"""Read declared ORIGINAL vehicle constants by AST; zero environment starts."""
import ast,json
from pathlib import Path
from project_paths import ROOT
from chapter3_bser.safe_rl.options import validate
from chapter3_bser.safe_rl.dynamics import speed_audit


def declared_specs(root=ROOT):
    tree=ast.parse((Path(root)/'core/env/uav_env.py').read_text(encoding='utf-8'))
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='agent_specs' for t in n.targets))
    fields={'name','a_xy_max','a_z_max','v_xy_max','v_z_max','drag_xy','drag_z','buoyancy_bias','sensor_range'}
    specs=[{kw.arg:ast.literal_eval(kw.value) for kw in item.keywords if kw.arg in fields} for item in assignment.value.elts]
    gain=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='flow_gain' for t in n.targets))
    if ast.literal_eval(gain)!=.18:raise ValueError('review flow assumptions after physical source change')
    return specs


def run():
    options=validate();contract=dict(dt=.2,obstacle_sensor_range=4.5)
    rows=speed_audit(declared_specs(),contract,options)
    return dict(schema='auv.safe_speed.audit.v1',environment_initializations=0,physics_steps=0,
        flow_component_bound=.18,dt=.2,reaction_steps=options['reaction_steps'],margin=.52,
        rows=rows,safety_guarantee=False,
        scope='Conservative drag-discarding reaction+discretization+stop audit, independent axes. Actual ray coverage may be less than4.5m; runtime uses current evidence and separately checks combined3D motion.')


def main():print(json.dumps(run(),indent=2))
if __name__=='__main__':main()
