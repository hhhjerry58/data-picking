import math
import numpy as np
import torch

from data_picking.gradient_patterns import summarize_gradients


def test_gradient_norm_max_cosine_and_tied_parameters():
    a = torch.nn.Parameter(torch.zeros(2, 2))
    b = torch.nn.Parameter(torch.zeros(2))
    a.grad = torch.tensor([[3., -4.], [0., 0.]])
    b.grad = torch.tensor([0., 12.])
    anchor = {"model.layers.0.proj.weight": torch.ones_like(a), "model.norm.weight": torch.ones_like(b)}
    named = [("model.layers.0.proj.weight", a), ("model.norm.weight", b), ("tied_duplicate", a)]
    result, layers, details = summarize_gradients(named, anchor, math.sqrt(6))
    assert result["parameter_count"] == 6
    assert result["grad_l2"] == 13
    assert result["grad_max_abs"] == 12
    assert result["grad_max_abs_parameter"] == "model.norm.weight"
    np.testing.assert_allclose(result["grad_rms"], 13/math.sqrt(6))
    np.testing.assert_allclose(result["cos_to_a1"], 11/(13*math.sqrt(6)))
    assert len(details) == 2
    assert {v["layer"]: v["grad_l2"] for v in layers} == {"layer_00":5., "final_norm":12.}


def test_per_example_norms_do_not_cancel_like_batch_gradient():
    p = torch.nn.Parameter(torch.tensor([0.]))
    norms = []
    for target in [-1., 1.]:
        p.grad = None
        (p-target).square().mean().backward()
        stats,_,_ = summarize_gradients([("weight",p)], {"weight":torch.ones_like(p)}, 1.)
        norms.append(stats["grad_l2"])
    p.grad = None
    torch.stack([(p-1).square(), (p+1).square()]).mean().backward()
    assert norms == [2.,2.]
    assert p.grad.item() == 0


