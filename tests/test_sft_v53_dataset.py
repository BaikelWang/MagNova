from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path
from magnova_agent.models import MaterialRequest

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"sft_v53"
ORDER=("id_hidden5","ood_hidden5","adversarial_hidden5","dev","train")
SIZES={"id_hidden5":300,"ood_hidden5":300,"adversarial_hidden5":300,"dev":400,"train":3200}

def load(split):
    rows=[json.loads(x) for x in (DATA/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    meta=[json.loads(x) for x in (DATA/f"{split}.meta.jsonl").read_text(encoding="utf-8").splitlines()]
    return [dict(a,**b) for a,b in zip(rows,meta,strict=True)]

def test_schema_sizes_and_isolation():
    originals=set(); inputs=set()
    for split in ORDER:
        rows=load(split); assert len(rows)==SIZES[split]
        for row in rows:
            assert row["original_user"] not in originals; originals.add(row["original_user"])
            user=row["conversations"][1]["content"]; assert user not in inputs; inputs.add(user)
            payload=json.loads(row["conversations"][2]["content"]); payload["original_text"]=row["original_user"]; MaterialRequest.model_validate(payload)

def test_hidden_alias_isolation():
    visible={a.casefold() for s in ("train","dev") for r in load(s) for a in r["property_aliases"].values()}
    hidden={a.casefold() for s in ("ood_hidden5","adversarial_hidden5") for r in load(s) for a in r["property_aliases"].values()}
    assert hidden and not visible.intersection(hidden)

def test_train_focus_counts_and_balance():
    families=Counter(r["case_family"] for r in load("train")); assert families=={"task_contrast":800,"double_range":600,"operator_pair":600,"eq_contrast":400,"rare_earth":400,"v52b_rehearsal":400}
    tasks=Counter(r["task"] for r in load("train")); assert tasks=={"generate_material":1600,"screen_candidates":1600}
    rare=[r for r in load("train") if r["case_family"]=="rare_earth"]
    values=Counter(json.loads(r["conversations"][2]["content"])["composition"]["exclude_rare_earth"] for r in rare); assert values=={True:200,False:200}

def test_manifest_hashes():
    m=json.loads((DATA/"manifest.json").read_text(encoding="utf-8")); assert m["generation_order"]==list(ORDER); assert m["frozen_before_training"]==list(ORDER[:3])
    for s in SIZES:
        assert hashlib.sha256((DATA/f"{s}.jsonl").read_bytes()).hexdigest()==m["splits"][s]["sha256"]
        assert hashlib.sha256((DATA/f"{s}.meta.jsonl").read_bytes()).hexdigest()==m["splits"][s]["metadata_sha256"]
