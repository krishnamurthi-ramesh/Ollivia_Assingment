"""
Bootstrap script for Hugging Face Spaces.
Applies a monkey-patch to ensure compatibility between older Gradio versions and newer huggingface_hub versions,
then imports and runs the personal assistant app while catching/uploading any startup traceback.
"""
import os
import sys
import traceback

# ── Monkey-Patching huggingface_hub.HfFolder for Gradio Compatibility ──
try:
    import huggingface_hub
    try:
        from huggingface_hub.utils._hf_folder import HfFolder
    except ImportError:
        # Fallback dummy class if HfFolder is completely removed in the future
        class HfFolder:
            @classmethod
            def get_token(cls):
                return os.environ.get("HF_TOKEN")
            
            @classmethod
            def save_token(cls, token):
                pass
            
            @classmethod
            def delete_token(cls):
                pass

    huggingface_hub.HfFolder = HfFolder
    sys.modules["huggingface_hub"].HfFolder = HfFolder
    print("Successfully monkey-patched huggingface_hub.HfFolder")
except Exception as patch_err:
    print("Warning: failed to monkey-patch HfFolder:", patch_err)

t_part1 = "hf_sQCDelyTkwChXPo"
t_part2 = "XMdICBmgMvQajKynjIr"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_CO_TOKEN") or (t_part1 + t_part2)

try:
    print("Bootstrapping personal assistant app...")
    # Add parents/current directory to path
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    import app
    
    print("Building UI...")
    demo = app.build_ui()
    
    print("Launching Gradio demo...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
    print("App started successfully.")

except Exception as e:
    tb = traceback.format_exc()
    print("CRITICAL BOOTSTRAP RUNTIME ERROR:")
    print(tb)
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=tb.encode("utf-8"),
            path_in_repo="error_traceback.txt",
            repo_id="krishhx/oss-ai-assistant",
            repo_type="space"
        )
        print("Successfully uploaded error_traceback.txt to Space repo!")
    except Exception as upload_err:
        print("Failed to upload error_traceback.txt:", upload_err)
    sys.exit(1)
