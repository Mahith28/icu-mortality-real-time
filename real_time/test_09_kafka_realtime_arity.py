import ast
from pathlib import Path

PATH = Path(__file__).resolve().with_name("09_kafka_realtime.py")
tree = ast.parse(PATH.read_text())
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
targets = {"process_observation", "attempt_final", "retry_due_finals", "evict_stay", "finalize_idle_stays", "publish_live_update"}
aliases = {("attempt_final", "window_id"): {"candidate_window_id"}, ("publish_live_update", "live_observations"): {"live_observations[stay_id]"}, ("publish_live_update", "source_type"): {"str(observation.get('source_type', 'unknown'))"}, ("publish_live_update", "event_id"): {"str(observation.get('event_id', ''))"}}
errors = []
for node in ast.walk(tree):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in targets:
        continue
    name = node.func.id
    function = functions[name]
    names = [a.arg for a in function.args.args]
    required_count = len(names) - len(function.args.defaults)
    if len(node.args) < required_count:
        errors.append(f"{name} line {node.lineno}: {len(node.args)} args, {required_count} required")
        continue
    if len(node.args) > len(names):
        errors.append(f"{name} line {node.lineno}: {len(node.args)} args, {len(names)} parameters")
        continue
    for i, arg in enumerate(node.args):
        passed = ast.unparse(arg)
        parameter = names[i]
        if passed != parameter and passed not in aliases.get((name, parameter), set()) and not (parameter == "observation" and passed == "data"):
            errors.append(f"{name} line {node.lineno}: arg {i} is '{passed}', parameter is '{parameter}'")
if errors:
    print("STRICT NAME-ALIGNED TEST FAILED")
    print("\n".join(errors))
    raise SystemExit(1)
print("STRICT NAME-ALIGNED TEST PASSED")
