from types import SimpleNamespace

import torch

from data_picking.document_utils import document_windows
from data_picking.loss import per_example_nll
from data_picking.tokenization import collate
from data_picking.web_gradients import backward_document


def test_document_gradient_weights_unequal_windows_by_scored_tokens(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(.2))
        def forward(self, input_ids, **kwargs):
            logits = torch.stack([self.p, -self.p]).expand(*input_ids.shape, 2)
            return SimpleNamespace(logits=logits)
    model = Model()
    ids = torch.tensor([[0, 1, 1, 1, 0]])
    _, means, _ = per_example_nll(model(ids).logits, ids, torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.bool))
    expected_loss = means.item()
    means[0].backward()
    expected_gradient = model.p.grad.clone()
    monkeypatch.setattr("data_picking.web_gradients.collate", lambda rows, pad: collate(rows, pad, device="cpu"))
    windows = list(document_windows("doc", ids[0].tolist(), 4))
    loss = backward_document(model, SimpleNamespace(pad_token_id=0), windows)
    torch.testing.assert_close(model.p.grad, expected_gradient)
    assert abs(loss-expected_loss) < 1e-6
