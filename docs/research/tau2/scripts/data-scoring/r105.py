from tau2.data_model.message import ToolCall
from tau2.registry import registry

env = registry.get_env_constructor("retail")()
db = env.tools.db
o = db.orders["#W4316152"]
u = db.users[o.user_id]
print("user", o.user_id, "status", o.status)
for pid, pm in u.payment_methods.items():
    print(" pm", pid, pm.model_dump())
items = {i.item_id: i.price for i in o.items}
print(" order items", [(i.item_id, i.price) for i in o.items])
for nid in ["3761330360", "9647374798"]:
    for p in db.products.values():
        if nid in p.variants:
            print(" new", nid, p.variants[nid].price, p.variants[nid].available)
task = [t for t in registry.get_tasks_loader("retail")(None) if t.id == "105"][0]
h0 = env.get_db_hash()
for a in task.evaluation_criteria.actions:
    r = env.get_response(
        ToolCall(id="1", name=a.name, arguments=a.arguments, requestor=a.requestor)
    )
    print("RESULT error=", r.error, r.content[:300])
print("db unchanged:", env.get_db_hash() == h0)
