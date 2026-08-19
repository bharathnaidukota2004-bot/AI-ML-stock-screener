import os
import zipfile
import shutil

def prepare_package():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = r"C:\Users\mail\.gemini\antigravity-ide\brain\67f4ad5a-317c-4b21-a114-d9879689b761"
    os.makedirs(artifacts_dir, exist_ok=True)

    zip_filename = "StockMarket_AIML_Screener_SourceCode.zip"
    zip_filepath = os.path.join(project_dir, zip_filename)
    zip_artifact_path = os.path.join(artifacts_dir, zip_filename)

    included_dirs = ["config", "core", "ml", "ui"]
    included_files = ["main.py", "build_exe.py", "requirements.txt", "README.md"]

    print("Creating clean Source Code ZIP archive...")
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in included_files:
            fp = os.path.join(project_dir, f)
            if os.path.exists(fp):
                zipf.write(fp, arcname=f)
                print(f"  Added file: {f}")

        for d in included_dirs:
            dp = os.path.join(project_dir, d)
            if os.path.exists(dp):
                for root, dirs, files in os.walk(dp):
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        if file.endswith(".pyc"):
                            continue
                        abs_p = os.path.join(root, file)
                        rel_p = os.path.relpath(abs_p, project_dir)
                        zipf.write(abs_p, arcname=rel_p)
                        print(f"  Added file: {rel_p}")

    print(f"Source Code Zip saved: {zip_filepath}")
    shutil.copy2(zip_filepath, zip_artifact_path)
    print(f"Copied to Artifacts: {zip_artifact_path}")

    # Copy Executable to Artifacts if present
    exe_src = os.path.join(project_dir, "dist", "StockMarket_AIML_Screener.exe")
    exe_artifact = os.path.join(artifacts_dir, "StockMarket_AIML_Screener.exe")
    if os.path.exists(exe_src):
        shutil.copy2(exe_src, exe_artifact)
        print(f"Executable copied to Artifacts: {exe_artifact}")

    print("Package preparation completed successfully.")

if __name__ == "__main__":
    prepare_package()
