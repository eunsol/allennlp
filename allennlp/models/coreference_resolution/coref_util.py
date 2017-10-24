import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from allennlp.nn import util

def custom_rnn(inputs: torch.Tensor, mask: torch.Tensor, rnn, hidden_state: torch.Tensor = None) -> torch.Tensor:
  # Copied from PytorchSeq2SeqWrapper class.
  # In some circumstances you may have sequences of zero length.
  # ``pack_padded_sequence`` requires all sequence lengths to be > 0, so here we
  # adjust the ``mask`` so that every sequence has length at least 1. Then after
  # running the RNN we zero out the corresponding rows in the result.
  # First count how many sequences are empty.
  batch_size = mask.size()[0]
  num_valid = torch.sum(mask[:, 0]).int().data[0]

  # Force every sequence to be length at least one. Need to `.clone()` the mask
  # to avoid a RuntimeError from shared storage.
  if num_valid < batch_size:
      mask = mask.clone()
      mask[:, 0] = 1
  sequence_lengths = util.get_lengths_from_binary_sequence_mask(mask)
  sorted_inputs, sorted_sequence_lengths, restoration_indices = util.sort_batch_by_length(inputs,
                                                                                     sequence_lengths)
  packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                               sorted_sequence_lengths.data.tolist(),
                                               batch_first=True)

  # Actually call the module on the sorted PackedSequence.
  packed_sequence_output, _ = rnn(packed_sequence_input, hidden_state)
  unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

  # We sorted by length, so if there are invalid rows that need to be zeroed out
  # they will be at the end.
  if num_valid < batch_size:
      unpacked_sequence_tensor[num_valid:, :, :] = 0.

  # Restore the original indices and return the sequence.
  return unpacked_sequence_tensor.index_select(0, restoration_indices)
