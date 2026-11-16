"""Resolve machine-specific inputs separately from the public snapshot."""
from pathlib import Path
import json
import os

EXPORT_ROOT=Path(__file__).resolve().parents[1]
CONFIG_FILE=os.environ.get('LTLF_PATHS')
CONFIG=json.loads(Path(CONFIG_FILE).read_text(encoding='utf-8-sig')) if CONFIG_FILE else {}

def configured(name,default):
    path=Path(CONFIG.get(name,default)).expanduser()
    return path.resolve() if path.is_absolute() else (EXPORT_ROOT/path).resolve()

AUDIT_ROOT=configured('audit_dir','private/outputs/research_audit')
WORK_DIR=configured('work_dir','private/work')
MODEL_DIR=EXPORT_ROOT/'models'
COVEO_DATA=configured('coveo_data','private/data/browsing_draw_random4_sessions_product_compact.csv')
COVEO_DIAGNOSTIC_DATA=configured('coveo_diagnostic_data','private/data/browsing_draw_random4_sessions.csv')
COVEO_BASIS=EXPORT_ROOT/'formulas/coveo_ranked20.json'
ELECTRONICS_BASIS=EXPORT_ROOT/'formulas/electronics_top10.json'
COSMETICS_BASIS=EXPORT_ROOT/'formulas/cosmetics_ranked20.json'

def resolve_saved_path(value):
    """Translate explicit path prefixes; never infer file identity by basename."""
    s=str(value).replace('\\','/')
    mappings=dict(CONFIG.get('path_mappings',{}))
    mappings.setdefault('<workspace>/work',str(WORK_DIR))
    mappings.setdefault('<workspace>/outputs/research_audit',str(AUDIT_ROOT))
    for old,new in sorted(mappings.items(),key=lambda item:len(item[0]),reverse=True):
        old=old.replace('\\','/').rstrip('/')
        if s==old or s.startswith(old+'/'):
            return Path(str(new)).expanduser()/s[len(old):].lstrip('/')
    return Path(value).expanduser()
