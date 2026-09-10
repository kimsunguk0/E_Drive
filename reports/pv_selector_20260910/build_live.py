import hashlib, pathlib
SRC = pathlib.Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/experiments/c_refine_20260910")
DST = pathlib.Path("experiments/pv_selector_20260910")
s = (SRC / "evaluate_c_scene_selector.py").read_text()

def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            d.update(c)
    return d.hexdigest()

new_sha = sha(DST / "pv_selector.py")
s = s.replace("SCENE_SOURCE_SHA = 'be8d9eef8e1e9f087852c1a81d0f5f4437e618a146d02d5037de60181b360ae5'",
              "SCENE_SOURCE_SHA = '%s'" % new_sha)

s = s.replace("require(schema in ('c_scene_selector_frozen_v1','c_scene_continuation_v1'),'Unknown refinement schema')",
              "require(schema in ('c_scene_selector_frozen_v1','pv_selector_status_v1','c_scene_continuation_v1'),'Unknown refinement schema')")
s = s.replace("    if schema=='c_scene_selector_frozen_v1':",
              "    if schema in ('c_scene_selector_frozen_v1','pv_selector_status_v1'):")

old = """    require(mode in ('real','zero'),'Invalid scene-head mode')
    require(list(p)==[128,20] and list(v)==[64,64],'Only the frozen selected V64 protocol is supported')
    return dict(path_filter=list(p),velocity_filter=list(v),mode=mode,original_c_checkpoint=public_c)"""
new = """    require(mode in ('real','zero'),'Invalid scene-head mode')
    # 'status' routes the provided causal vx,vy,ax,ay into the 32 selection
    # features of already-completed bank candidates; absent means the old zeros.
    status=manifest.get('arguments',{}).get('status','zero')
    require(status in ('real','zero'),'Invalid selection status mode')
    require(list(p)==[128,20] and list(v)==[64,64],'Only the frozen selected V64 protocol is supported')
    return dict(path_filter=list(p),velocity_filter=list(v),mode=mode,status=status,
                original_c_checkpoint=public_c)"""
assert old in s; s = s.replace(old, new)

s = s.replace("require(m['source_sha256'].get('c_scene_selector.py')==SCENE_SOURCE_SHA,'Unsupported scene head implementation')",
              "require(m['source_sha256'].get('pv_selector.py')==SCENE_SOURCE_SHA,'Unsupported scene head implementation')")
s = s.replace("mod=module_from(scene['source']/'c_scene_selector.py','_scene_head_recorded')",
              "mod=module_from(scene['source']/'pv_selector.py','_scene_head_recorded')")

old = """    def __init__(self, original_c, scene_head, scene_module):
        super().__init__()
        self.capture=scene_module.CandidateTokenCapture(original_c,detach_tokens=True,freeze_base=True)
        self.scene_head=scene_head"""
new = """    def __init__(self, original_c, scene_head, scene_module, status_mode='zero'):
        super().__init__()
        self.capture=scene_module.CandidateTokenCapture(original_c,detach_tokens=True,freeze_base=True)
        self.scene_head=scene_head
        if status_mode not in ('real','zero'):
            raise ValueError('status_mode must be real or zero')
        self.status_mode=status_mode

    @staticmethod
    def status8_from(inputs, status_mode):
        \"\"\"Provided causal vx,vy,ax,ay for final selection only; None when zero.

        The same tensor already conditions the shared perception query, so this
        reads it from the live batch rather than from any offline artifact.
        \"\"\"
        if status_mode!='real':
            return None
        state=inputs['perception_status']
        if state.shape[-1:]!=(4,) or not bool(torch.isfinite(state).all()):
            raise ValueError('perception_status must be finite [...,4]')
        status8=state.new_zeros((state.shape[0],8))
        status8[:,4:8]=state
        return status8"""
assert old in s; s = s.replace(old, new)

old = """    def forward(self, **inputs):
        output=self.capture(**inputs)
        return self.scene_head(output,goal_xy=inputs['goal_xy'])"""
new = """    def forward(self, **inputs):
        output=self.capture(**inputs)
        return self.scene_head(output,goal_xy=inputs['goal_xy'],
                               status=self.status8_from(inputs,self.status_mode))"""
assert old in s; s = s.replace(old, new)

s = s.replace("    model=FullSceneModel(original,head,mod).eval()",
              "    model=FullSceneModel(original,head,mod,scene['config']['status']).eval()")

old = "    def compare_batch(self, batch, output, goal, head, bank, start):"
new = "    def compare_batch(self, batch, output, goal, head, bank, start, status=None):"
assert old in s; s = s.replace(old, new)
s = s.replace("        repeated=head(offline,goal_xy=refs['goal_xy'])",
              "        repeated=head(offline,goal_xy=refs['goal_xy'],status=status)")
s = s.replace("                cache.compare_batch(batch,out,inputs['goal_xy'],head,bank,start)",
              "                cache.compare_batch(batch,out,inputs['goal_xy'],head,bank,start,\n"
              "                                    FullSceneModel.status8_from(inputs,model.status_mode))")

s = s.replace("require(scene['manifest']['schema']=='c_scene_selector_frozen_v1','A frozen base cache cannot audit a continued model_c')",
              "require(scene['manifest']['schema'] in ('c_scene_selector_frozen_v1','pv_selector_status_v1'),'A frozen base cache cannot audit a continued model_c')")
s = s.replace("new_head_signature='output, goal_xy; no raw or predicted status input'",
              "new_head_signature='output, goal_xy, provided causal status for completed-candidate selection only'")

(DST / "evaluate_pv_status_selector.py").write_text(s)
print("wrote", DST / "evaluate_pv_status_selector.py", "scene_sha", new_sha[:16])
