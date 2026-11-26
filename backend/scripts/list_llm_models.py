"""Lists models your keys can use and tests one call per provider. Run: python -m scripts.list_llm_models"""
import os
from openai import OpenAI
from src.config import PROVIDERS
def main():
    for prov, cfg in PROVIDERS.items():
        key = os.getenv(cfg["key_env"], "") if cfg["key_env"] else "ollama"
        if not key:
            print(f"\n== {prov}: no key set, skipped")
            continue
        print(f"\n== {prov} (configured model: {cfg['model']})")
        try:
            c = OpenAI(base_url=cfg["base_url"], api_key=key, timeout=20, max_retries=0)
            ids = sorted(m.id for m in c.models.list())

            interesting = [i for i in ids if any(s in i for s in ("flash", "gpt-oss", "llama", "qwen", "gemma"))]
            print("   available (filtered):", ", ".join(interesting[:40]) or ", ".join(ids[:40]))
            print("   configured model listed:", any(i.endswith(cfg["model"]) for i in ids))
            r = c.chat.completions.create(model=cfg["model"], max_tokens=256, temperature=0,
                                          messages=[{"role": "user", "content": "Reply with the single word OK"}])
            print("   test call ->", (r.choices[0].message.content or "").strip()[:50])
        except Exception as e:  # noqa: BLE001
            print("   ERROR:", str(e)[:300])
if __name__ == "__main__":
    main()
