import pathlib, shutil
SRC = pathlib.Path("/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/experiments/c_refine_20260910")
DST = pathlib.Path("experiments/pv_selector_20260910")
DST.mkdir(parents=True, exist_ok=True)
s = (SRC / "cache_c_candidates.py").read_text()

anchor = "def exact_token_dtype(token):"
assert anchor in s
donor = '''_DONOR = {"path": None, "sha256": None}


class DonorImageDataset(torch.utils.data.Dataset):
    """Recipient row with another scene's images; everything else untouched.

    Only ``images`` and ``history_images`` are taken from the donor. The
    recipient keeps its own calibration (``lidar2img``), ``image_hw``,
    ``time_offsets``, causal status, goal, row identity and GT, matching the
    primary model's different-scene substitution control.
    """

    def __init__(self, base, donor_index, rows):
        self.base = base
        self.donor_index = np.asarray(donor_index, dtype=np.int64)
        self.rows = rows
        if len(self.donor_index) != len(base):
            raise ValueError("Donor index must cover every dataset position")

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name in ("base", "donor_index", "rows"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getitem__(self, index):
        item = dict(self.base[index])
        donor = self.base[int(self.donor_index[index])]
        item["images"] = donor["images"]
        item["history_images"] = donor["history_images"]
        return item


def _wrap_donor_images(dataset, rows):
    mapping = json.loads(Path(_DONOR["path"]).read_text())
    rows = np.asarray(rows, dtype=np.int64)
    position = {int(row): i for i, row in enumerate(rows)}
    if len(mapping) != len(rows):
        raise ValueError("Donor mapping must cover exactly the evaluated rows")
    if {int(r["recipient_row"]) for r in mapping} != set(position):
        raise ValueError("Donor mapping recipient rows differ from the dataset")
    donor_index = np.empty(len(rows), dtype=np.int64)
    for record in mapping:
        recipient, image_row = int(record["recipient_row"]), int(record["image_row"])
        if image_row not in position:
            raise ValueError("Donor image row is outside the evaluated population")
        if record["recipient_scene"] == record["image_scene"]:
            raise ValueError("Donor must come from a different scene")
        donor_index[position[recipient]] = position[image_row]
    if bool((donor_index == np.arange(len(rows))).any()):
        raise ValueError("A row would donate its own images")
    return DonorImageDataset(dataset, donor_index, rows)


'''
s = s.replace(anchor, donor + anchor)

old = """    return dataset, provenance, identities"""
new = """    if _DONOR["path"] is not None:
        if limit:
            raise ValueError("Donor substitution runs on the full population only")
        dataset = _wrap_donor_images(dataset, identities["rows"])
        provenance.update(image_substitution={
            "mode": "fixed different-scene donor images",
            "replaced_keys": ["images", "history_images"],
            "unchanged": ["lidar2img", "image_hw", "time_offsets", "causal status",
                          "goal", "row identity", "GT"],
            "mapping_path": str(Path(_DONOR["path"]).resolve()),
            "mapping_sha256": _DONOR["sha256"]})
    return dataset, provenance, identities"""
assert old in s
s = s.replace(old, new, 1)

old = "    p.add_argument('--gpu', type=int, choices=(1, 4))"
new = ("    p.add_argument('--gpu', type=int, choices=(0, 1, 4))\n"
       "    p.add_argument('--image-donor-mapping', default=None,\n"
       "                   help='different-scene image substitution control')")
assert old in s
s = s.replace(old, new)

old = "def main():"
new = '''def _install_donor(path):
    if path is None:
        return
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    _DONOR.update(path=path, sha256=digest.hexdigest())


def main():'''
assert old in s
s = s.replace(old, new, 1)

marker = "    a = arguments()"
assert marker in s
s = s.replace(marker, marker + "\n    _install_donor(a.image_donor_mapping)", 1)

gate = "    require(a.audit_only or a.gpu in (1, 4), 'Cache task is assigned GPU1/train or GPU4/tune')"
assert gate in s
s = s.replace(gate, "    require(a.audit_only or a.gpu in (0, 1, 4), 'Cache task runs on an allocated GPU')")

(DST / "cache_c_candidates_ablation.py").write_text(s)
for name in ("c_scene_selector.py",):
    shutil.copy2(SRC / name, DST / name)
print("wrote", DST / "cache_c_candidates_ablation.py")
