
SEQ_LENGTH = 5


def split_sentence(input_data):
  assert isinstance(input_data[0], list)
  # Load data
  datasets = []
  for line in input_data:
      for i in range(1, SEQ_LENGTH+1):
          data = line[:i] + [5000] * (SEQ_LENGTH-i)
          datasets.append(data)
  return datasets


if __name__ == "__main__":

  s = [[112, 234, 54, 6546,342],[232, 455, 6657, 4342, 2342]]

  split_s = split_sentence(s)

  print(split_s)