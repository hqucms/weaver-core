"""Regression checks for tuple outputs with aligned labels and masks."""

import unittest

import torch

from weaver.utils.nn.tools import _flatten_preds


class FlattenTuplePredictionsTest(unittest.TestCase):
    def test_two_tuple_uses_returned_mask(self):
        preds = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        labels = torch.arange(8).reshape(2, 4)
        mask = torch.tensor([[True, False, True, False], [False, True, True, False]])
        logits, targets, returned_mask = _flatten_preds((preds, mask), label=labels)
        torch.testing.assert_close(logits, preds.transpose(1, 2)[mask])
        torch.testing.assert_close(targets, labels[mask])
        self.assertIs(returned_mask, mask)

    def test_three_tuple_uses_returned_labels_and_mask(self):
        preds = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        labels = torch.arange(8).reshape(2, 4)
        mask = torch.tensor([[True, False, True, False], [False, True, True, False]])
        logits, targets, returned_mask = _flatten_preds(
            (preds, labels, mask), label=torch.full_like(labels, -1), mask=torch.ones_like(mask)
        )
        torch.testing.assert_close(logits, preds.transpose(1, 2)[mask])
        torch.testing.assert_close(targets, labels[mask])
        self.assertIs(returned_mask, mask)


if __name__ == "__main__":
    unittest.main()
