# Optional check tool (not used by train.sh): run stages 1-2 on a 24 GB GPU.
# The trainer accumulates gradients over microbatches with whole-batch loss normalizers,
# so a smaller microbatch leaves the logical batch (16) and the math unchanged.
# The declared recipe and the parsed runtime are both set to the same microbatch, and the
# ~174 GiB free-memory pre-flight (cannot pass on 24 GB) is disabled.
import argparse, importlib.abc, importlib.machinery, os, sys
MB = os.environ.get("RELCHECK_MICROBATCH")
if MB:
    MB = int(MB)
    _orig = argparse.ArgumentParser.parse_known_args
    def parse_known_args(self, *a, **k):
        ns, rest = _orig(self, *a, **k)
        if hasattr(ns, "microbatch") and hasattr(ns, "cuda_memory_limit_mib"):
            print(f"[relcheck] runtime microbatch {ns.microbatch}->{MB}, cuda limit {ns.cuda_memory_limit_mib}->0", flush=True)
            ns.microbatch, ns.cuda_memory_limit_mib, ns.cuda_min_free_mib = MB, 0, 0
        return ns, rest
    argparse.ArgumentParser.parse_known_args = parse_known_args

    def patch_recipe(experiment):
        r = (experiment or {}).get("recipe")
        if isinstance(r, dict):
            print("[relcheck] declared microbatch", r.get("microbatch"), "->", MB, flush=True)
            r["microbatch"] = MB
            for key in ("cuda_memory_limit_mib", "cuda_min_free_mib"):
                if key in r: r[key] = 0

    class Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[-1] != "train_motiondrive_v2":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None: return None
            loader = spec.loader; orig_exec = loader.exec_module
            def exec_module(module):
                orig_exec(module)
                run = module.run_training
                def run_training(*a, **k):
                    patch_recipe(k.get("experiment"))
                    for m in list(sys.modules.values()):
                        if getattr(m, "MICROBATCH", None) not in (None, MB) and isinstance(getattr(m, "MICROBATCH"), int):
                            print("[relcheck] module", m.__name__, "MICROBATCH", m.MICROBATCH, "->", MB, flush=True); m.MICROBATCH = MB
                    return run(*a, **k)
                module.run_training = run_training
            loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, Hook())
