import os
import json
from typing import Dict, Any
import requests


def _heuristic_decision(feature: Dict[str, Any]) -> Dict[str, Any]:
    ftype = feature.get("type")
    missing = feature.get("missing_rate", 0.0)
    skew = feature.get("skew", 0.0)
    cardinality = feature.get("cardinality", 0)

    if ftype == "numeric":
        if missing > 0.2:
            method = "median"
            reason = "Taux de valeurs manquantes élevé; la médiane est robuste."
        else:
            method = "mean" if abs(skew) < 1.0 else "median"
            reason = "Skew faible -> moyenne; sinon médiane plus robuste."
        scaling = "standard" if abs(skew) < 1.0 else "minmax"
        return {"imputation": method, "encoding": None, "scaling": scaling, "reason": reason}

    if ftype == "categorical":
        if cardinality <= 20:
            enc = "onehot"
            reason = "Faible cardinalité -> One-Hot."
        else:
            enc = "frequency"
            reason = "Haute cardinalité -> Frequency encoding."
        return {"imputation": "most_frequent", "encoding": enc, "scaling": None, "reason": reason}

    if ftype == "datetime":
        return {"imputation": "drop" if missing > 0.3 else "leave", "encoding": "cyclical", "scaling": None, "reason": "Dates encodées cycliquement si gardées."}

    return {"imputation": "leave", "encoding": None, "scaling": None, "reason": "Type non pris en charge, laissé tel quel."}


def _llm_endpoint_config():
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("HUAWEI_LLM_API_KEY")
    )
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE")
    full_url = os.getenv("FULL_CHAT_COMPLETIONS_URL") or os.getenv("HUAWEI_LLM_ENDPOINT")
    auth_header = os.getenv("LLM_AUTH_HEADER", "Authorization").strip()
    if not api_key:
        return None, None, None, None
    # Prefer full URL if provided
    if full_url:
        return None, api_key, full_url.strip(), auth_header
    if not base_url:
        return None, None, None, None
    base = base_url.rstrip('/')
    # If user already provided the full path including /chat/completions, keep it
    if base.endswith('/chat/completions'):
        return None, api_key, base, auth_header
    return base, api_key, f"{base}/chat/completions", auth_header


def advise(feature_stats: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    """
    feature_stats example:
    {"name": "Age", "type": "numeric", "missing_rate": 0.12, "skew": 1.8, "cardinality": 0}
    Returns decisions: {imputation, encoding, scaling, reason}
    """
    base_url, api_key, final_url, auth_header = _llm_endpoint_config()
    if not api_key or not final_url:
        return _heuristic_decision(feature_stats)

    system = (
        "Tu es un expert en preprocessing. Donne des recommandations concises en JSON; "
        "clés: imputation (mean|median|knn|drop|leave|most_frequent), encoding (onehot|frequency|target|cyclical|none), "
        "scaling (standard|minmax|none), reason (français court)."
    )
    user = (
        "Voici les statistiques d'une colonne. Propose la meilleure stratégie.\n" +
        json.dumps(feature_stats, ensure_ascii=False)
    )

    # Build headers: default Authorization: Bearer; allow X-Auth-Token per Huawei guide
    headers = {"Content-Type": "application/json"}
    if auth_header.lower() == "x-auth-token":
        headers["X-Auth-Token"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": (model or "qwen3-32b"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(final_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        # Expect OpenAI-compatible structure
        content = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        data = None
        try:
            data = json.loads(content)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                data = json.loads(m.group(0))
        if not isinstance(data, dict):
            return _heuristic_decision(feature_stats)
        data.setdefault("imputation", "leave")
        data.setdefault("encoding", None)
        data.setdefault("scaling", None)
        data.setdefault("reason", "")
        return data
    except Exception:
        return _heuristic_decision(feature_stats)
