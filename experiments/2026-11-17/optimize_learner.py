"""Combine additive gradient contributions into one backward, with gradient check."""
from common import *
import torch,copy,importlib.util,time

def patch(repo):
    path=repo/'train.py';src=path.read_text(encoding='utf-8');tree_=ast.parse(src)
    class Fuse(ast.NodeTransformer):
        def visit_Expr(self,node):
            call=node.value
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='backward':
                expr=call.func.value
                value=expr if isinstance(expr,ast.Name) and expr.id=='loss' else ast.BinOp(left=ast.Name(id='total_loss',ctx=ast.Load()),op=ast.Add(),right=expr)
                return ast.Assign(targets=[ast.Name(id='total_loss',ctx=ast.Store())],value=value)
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='step':
                return [ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='total_loss',ctx=ast.Load()),attr='backward',ctx=ast.Load()),args=[],keywords=[])),node]
            return node
    result=ast.fix_missing_locations(Fuse().visit(tree_));path.write_text(ast.unparse(result),encoding='utf-8')

def validate(repo):
    spec=importlib.util.spec_from_file_location('gradient_model',repo/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch.manual_seed(42);torch.set_num_threads(1);m.device=torch.device('cpu')
    a=m.Net(10,19);b=copy.deepcopy(a);x=torch.rand(7,10,19);y=torch.rand(7,10,1);mask=torch.zeros(7,10,1);mask[:,0]=1
    def terms(model):
        prediction=model(x);terms=[torch.sum((prediction[:,:,:1]*mask-y)**2)]
        ta=model.chose_right_.transpose(0,1);tb=model.op_chose_weight.transpose(0,1)
        for i in range(1,10):terms.append(.1*(ta[i].sum()+tb[i-1].sum()-1)**2)
        terms.append(torch.sum(.1*torch.relu(model.op_chose_weight[0][:9]-model.op_chose_weight[0][1:])))
        for i in range(8):
            for j in range(i+2,10):terms.append(torch.sum(torch.relu(model.chose_right_[i][j]+model.chose_right_[i+1:j][j+1:10]-1)*.1))
        return terms
    for term in terms(a):term.backward(retain_graph=True)
    sum(terms(b)).backward();diff=[]
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad,pb.grad,atol=2e-5,rtol=2e-5);diff.append(float((pa.grad-pb.grad).abs().max()))
    return dict(status='passed',max_abs_gradient_difference=max(diff),atol=2e-5,rtol=2e-5,change='sum identical additive loss terms before one backward; no loss or optimizer setting changed')
