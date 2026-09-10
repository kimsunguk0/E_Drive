import pathlib
p = pathlib.Path("experiments/pv_selector_20260910/train_pv_selector.py")
s = p.read_text()
old = "    p.add_argument('--status', choices=('real', 'zero'), required=True)"
new = (old + "\n"
       "    # 'zero' also removes the provided goal from the learned features, so the\n"
       "    # head learns only from image tokens, candidate geometry and base score.\n"
       "    p.add_argument('--goal', choices=('real', 'zero'), default='real')")
assert old in s
s = s.replace(old, new)

old = """    def inputs(self, index):
        ids = self.field('candidate_ids', index).long()"""
new = """    def inputs(self, index):
        ids = self.field('candidate_ids', index).long()
        if self.drop_goal:
            return dict(candidate_ids=ids, candidate_xy=self.bank[ids],
                        candidate_valid=self.field('candidate_valid', index).bool(),
                        scores=self.field('scores', index).float(),
                        candidate_tokens=self.field('token', index)), None"""
assert old in s
s = s.replace(old, new)

old = """        self.status8 = None
        self.status_provenance = None"""
new = old + """
        self.drop_goal = False"""
assert old in s
s = s.replace(old, new)

old = """        if a.status == 'real':
            train.attach_status8('train')
            tune.attach_status8('tune')"""
new = """        if a.goal == 'zero':
            train.drop_goal = tune.drop_goal = True
        if a.status == 'real':
            train.attach_status8('train')
            tune.attach_status8('tune')"""
assert old in s
s = s.replace(old, new)

old = "                        goal='final completed-candidate selection only',"
new = ("                        goal=('final completed-candidate selection only'\n"
       "                              if a.goal == 'real' else 'absent from the learned head'),")
assert old in s
s = s.replace(old, new)
p.write_text(s)
print("patched")
