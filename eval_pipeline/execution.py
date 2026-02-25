"""
Sandboxed execution of LLM-generated Blender Python code.
"""

import os
import hashlib
import logging
import subprocess
import textwrap
from typing import Optional, Tuple

from eval_pipeline.config import Config

log = logging.getLogger(__name__)


_SAFETY_PREAMBLE = textwrap.dedent("""\
import sys
import bpy

# Safety: ensure we start clean
try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
except:
    pass

# ===== BEGIN GENERATED CODE =====
""")

_SAFETY_POSTAMBLE_TEMPLATE = textwrap.dedent("""
# ===== END GENERATED CODE =====

# Safety: verify output exists
import os as _os
if _os.path.exists('{expected_output}'):
    print("GENERATION_COMPLETE")
else:
    # Attempt emergency export of whatever is in the scene
    try:
        mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
        if mesh_objects:
            bpy.ops.object.select_all(action='DESELECT')
            for o in mesh_objects:
                o.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            bpy.ops.wm.obj_export(
                filepath='{expected_output}',
                export_selected_objects=True,
                export_uv=False,
                export_materials=False,
            )
            print("GENERATION_COMPLETE (emergency export)")
        else:
            print("GENERATION_FAILED: No mesh objects in scene")
    except Exception as e:
        print(f"GENERATION_FAILED: {{e}}")
""")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: EXECUTE GENERATED CODE
# ══════════════════════════════════════════════════════════════════════════════

def execute_generated_code(
    code: str,
    config: Config,
    work_dir: str,
) -> Tuple[bool, Optional[str], str]:
    """
    Execute LLM-generated Blender Python code in a sandboxed subprocess.

    The generated code is wrapped with a safety preamble (clean scene) and
    postamble (emergency export fallback) to recover partial successes.

    Returns:
        (success: bool, mesh_path: str|None, log_output: str)
    """
    unique_id = hashlib.md5(work_dir.encode()).hexdigest()[:12]
    expected_output = f"/tmp/generated_mesh_{unique_id}.obj"

    modified_code = code.replace("/tmp/generated_mesh.obj", expected_output)

    postamble = _SAFETY_POSTAMBLE_TEMPLATE.replace('{expected_output}', expected_output)
    wrapped_code = _SAFETY_PREAMBLE + modified_code + postamble

    code_path = os.path.abspath(os.path.join(work_dir, "generated_script.py"))
    with open(code_path, "w") as f:
        f.write(wrapped_code)

    if os.path.exists(expected_output):
        os.remove(expected_output)

    try:
        result = subprocess.run(
            [config.blender_path, "--background", "--python", code_path],
            capture_output=True,
            text=True,
            timeout=config.code_timeout,
            cwd=work_dir,
        )

        output = result.stdout + "\n" + result.stderr
        success = (
            result.returncode == 0
            and os.path.exists(expected_output)
        )

        if not success and os.path.exists(expected_output):
            success = True

        if success:
            final_path = os.path.join(work_dir, "generated_mesh.obj")
            os.rename(expected_output, final_path)
            return True, final_path, output
        else:
            log.debug(f"  Code execution failed. Last 300 chars:\n{output[-300:]}")
            return False, None, output

    except subprocess.TimeoutExpired:
        return False, None, f"TIMEOUT after {config.code_timeout}s"
    except Exception as e:
        return False, None, str(e)
