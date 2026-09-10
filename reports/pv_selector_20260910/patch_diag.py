import pathlib
src = pathlib.Path("experiments/c_refine_20260910/diagnose_c_velocity_stages.py")
dst = pathlib.Path("experiments/pv_selector_20260910/diagnose_velocity_stages_any.py")
s = src.read_text()
old = ("    require(plan.manifest['arguments']['common_status'] is True and "
       "plan.manifest['arguments']['history_mode']=='real','Expected temporal C')")
new = ("    # Same decomposition for the B arm, whose perception carries no status.\n"
       "    require(plan.manifest['arguments']['history_mode']=='real',"
       "'Expected a real-history temporal arm')\n"
       "    arm_common_status=plan.manifest['arguments']['common_status']\n"
       "    require(isinstance(arm_common_status,bool),'common_status must be an explicit bool')\n"
       "    print(json.dumps({'arm_common_status':arm_common_status}),flush=True)")
assert old in s, "anchor not found"
dst.write_text(s.replace(old, new))
print("wrote", dst)
