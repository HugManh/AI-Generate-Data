from unstructured.partition.auto import partition
import logging
logging.basicConfig(level=logging.DEBUG)
elements = partition(filename="test.txt")
print("------")
print(elements)
