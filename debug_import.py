try:
    print("Importing app...", flush=True)
    import src.ace_rm.app
    print("Import successful", flush=True)
except Exception as e:
    print(f"Import failed: {e}", flush=True)
