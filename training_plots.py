import re
import matplotlib.pyplot as plt

def parse_training_logs(logs):
    """Parses training logs and extracts accuracy and loss data.

    Args:
        logs (str): The raw training log output.

    Returns:
        dict: A dictionary containing lists of training and validation
              accuracies and losses.
              
              {"train_accs": [], "train_losses": [], "val_accs": [], "val_losses": []}
    """
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    # Regular expression to find the relevant data in each line.
    log_pattern = re.compile(
        r"accuracy: (\d+\.\d+) - loss: (\d+\.\d+) - val_accuracy: (\d+\.\d+) - val_loss: (\d+\.\d+)"
    )
    
    for line in logs.splitlines():
      match = log_pattern.search(line)
      if match:
          train_accs.append(float(match.group(1)))
          train_losses.append(float(match.group(2)))
          val_accs.append(float(match.group(3)))
          val_losses.append(float(match.group(4)))
    
    return {
        "train_accs": train_accs,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_losses": val_losses
    }


def plot_training_curves(data):
  """Plots the training and validation accuracy and loss.

      Args:
          data (dict):  A dictionary containing lists of training and validation
              accuracies and losses.

  """
  epochs = range(1, len(data["train_accs"]) + 1) # from 1 since epoch starts at 1

  plt.figure(figsize=(12, 6))

  # Plotting accuracy
  plt.subplot(1, 2, 1)
  plt.plot(epochs, data["train_accs"], 'r', label="Training accuracy")
  plt.plot(epochs, data["val_accs"], 'b', label="Validation accuracy")
  plt.title("Training and Validation Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()


  # Plotting loss
  plt.subplot(1, 2, 2)
  plt.plot(epochs, data["train_losses"], 'r', label="Training loss")
  plt.plot(epochs, data["val_losses"], 'b', label="Validation loss")
  plt.title("Training and Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()

  plt.tight_layout()
  plt.show()


# Your provided log output
logs = """
Epoch 1/50
  3/810 ━━━━━━━━━━━━━━━━━━━━ 32s 40ms/step - accuracy: 0.0061 - loss: 4.0735       
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1735600852.474275   22322 asm_compiler.cc:369] ptxas warning : Registers are spilled to local memory in function 'input_add_reduce_fusion_13', 4 bytes spill stores, 4 bytes spill loads

810/810 ━━━━━━━━━━━━━━━━━━━━ 59s 51ms/step - accuracy: 0.0417 - loss: 3.4341 - val_accuracy: 0.1024 - val_loss: 3.1144 - learning_rate: 0.0010
Epoch 2/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 32s 39ms/step - accuracy: 0.0924 - loss: 3.1646 - val_accuracy: 0.1963 - val_loss: 2.7490 - learning_rate: 0.0010
Epoch 3/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 32s 39ms/step - accuracy: 0.1786 - loss: 2.7683 - val_accuracy: 0.3495 - val_loss: 2.0893 - learning_rate: 0.0010
Epoch 4/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.3377 - loss: 2.1819 - val_accuracy: 0.6601 - val_loss: 1.2218 - learning_rate: 0.0010
Epoch 5/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.5337 - loss: 1.5340 - val_accuracy: 0.8014 - val_loss: 0.6884 - learning_rate: 0.0010
Epoch 6/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.6690 - loss: 1.1056 - val_accuracy: 0.8609 - val_loss: 0.4765 - learning_rate: 0.0010
Epoch 7/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.7287 - loss: 0.9024 - val_accuracy: 0.8807 - val_loss: 0.3941 - learning_rate: 0.0010
Epoch 8/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.7727 - loss: 0.7550 - val_accuracy: 0.9146 - val_loss: 0.3000 - learning_rate: 0.0010
Epoch 9/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.8174 - loss: 0.6158 - val_accuracy: 0.9261 - val_loss: 0.2601 - learning_rate: 0.0010
Epoch 10/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 32s 39ms/step - accuracy: 0.8412 - loss: 0.5249 - val_accuracy: 0.9321 - val_loss: 0.2235 - learning_rate: 0.0010
Epoch 11/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 32s 40ms/step - accuracy: 0.8491 - loss: 0.5080 - val_accuracy: 0.9505 - val_loss: 0.1750 - learning_rate: 0.0010
Epoch 12/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 32s 39ms/step - accuracy: 0.8682 - loss: 0.4341 - val_accuracy: 0.9527 - val_loss: 0.1644 - learning_rate: 0.0010
Epoch 13/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 32s 39ms/step - accuracy: 0.8811 - loss: 0.4057 - val_accuracy: 0.9547 - val_loss: 0.1554 - learning_rate: 0.0010
Epoch 14/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.8882 - loss: 0.3756 - val_accuracy: 0.9558 - val_loss: 0.1564 - learning_rate: 0.0010
Epoch 15/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.8990 - loss: 0.3443 - val_accuracy: 0.9534 - val_loss: 0.1556 - learning_rate: 0.0010
Epoch 16/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9013 - loss: 0.3288 - val_accuracy: 0.9584 - val_loss: 0.1455 - learning_rate: 0.0010
Epoch 17/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9068 - loss: 0.3148 - val_accuracy: 0.9531 - val_loss: 0.1633 - learning_rate: 0.0010
Epoch 18/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9066 - loss: 0.3153 - val_accuracy: 0.9604 - val_loss: 0.1375 - learning_rate: 0.0010
Epoch 19/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9086 - loss: 0.3146 - val_accuracy: 0.9636 - val_loss: 0.1247 - learning_rate: 0.0010
Epoch 20/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9211 - loss: 0.2578 - val_accuracy: 0.9667 - val_loss: 0.1178 - learning_rate: 0.0010
Epoch 21/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9207 - loss: 0.2672 - val_accuracy: 0.9633 - val_loss: 0.1292 - learning_rate: 0.0010
Epoch 22/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9132 - loss: 0.2892 - val_accuracy: 0.9703 - val_loss: 0.1119 - learning_rate: 0.0010
Epoch 23/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9237 - loss: 0.2582 - val_accuracy: 0.9675 - val_loss: 0.1177 - learning_rate: 0.0010
Epoch 24/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9275 - loss: 0.2364 - val_accuracy: 0.9652 - val_loss: 0.1259 - learning_rate: 0.0010
Epoch 25/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9213 - loss: 0.2575 - val_accuracy: 0.9692 - val_loss: 0.1098 - learning_rate: 0.0010
Epoch 26/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9348 - loss: 0.2149 - val_accuracy: 0.9697 - val_loss: 0.1114 - learning_rate: 0.0010
Epoch 27/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9349 - loss: 0.2111 - val_accuracy: 0.9691 - val_loss: 0.1162 - learning_rate: 0.0010
Epoch 28/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9358 - loss: 0.2128 - val_accuracy: 0.9698 - val_loss: 0.1089 - learning_rate: 0.0010
Epoch 29/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9384 - loss: 0.2052 - val_accuracy: 0.9683 - val_loss: 0.1117 - learning_rate: 0.0010
Epoch 30/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9365 - loss: 0.2057 - val_accuracy: 0.9711 - val_loss: 0.1057 - learning_rate: 0.0010
Epoch 31/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9405 - loss: 0.1972 - val_accuracy: 0.9718 - val_loss: 0.1048 - learning_rate: 0.0010
Epoch 32/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 33s 40ms/step - accuracy: 0.9426 - loss: 0.1892 - val_accuracy: 0.9729 - val_loss: 0.1043 - learning_rate: 0.0010
Epoch 33/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 32s 40ms/step - accuracy: 0.9407 - loss: 0.1971 - val_accuracy: 0.9711 - val_loss: 0.1056 - learning_rate: 0.0010
Epoch 34/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9461 - loss: 0.1750 - val_accuracy: 0.9723 - val_loss: 0.1079 - learning_rate: 0.0010
Epoch 35/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9439 - loss: 0.1842 - val_accuracy: 0.9740 - val_loss: 0.0989 - learning_rate: 0.0010
Epoch 36/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9434 - loss: 0.1862 - val_accuracy: 0.9712 - val_loss: 0.1038 - learning_rate: 0.0010
Epoch 37/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9452 - loss: 0.1813 - val_accuracy: 0.9714 - val_loss: 0.1063 - learning_rate: 0.0010
Epoch 38/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9435 - loss: 0.1876 - val_accuracy: 0.9717 - val_loss: 0.1055 - learning_rate: 0.0010
Epoch 39/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9460 - loss: 0.1799 - val_accuracy: 0.9740 - val_loss: 0.1033 - learning_rate: 0.0010
Epoch 40/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9501 - loss: 0.1682 - val_accuracy: 0.9735 - val_loss: 0.0981 - learning_rate: 0.0010
Epoch 41/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9496 - loss: 0.1666 - val_accuracy: 0.9739 - val_loss: 0.1029 - learning_rate: 0.0010
Epoch 42/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9492 - loss: 0.1653 - val_accuracy: 0.9705 - val_loss: 0.1110 - learning_rate: 0.0010
Epoch 43/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9420 - loss: 0.1920 - val_accuracy: 0.9742 - val_loss: 0.0994 - learning_rate: 0.0010
Epoch 44/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9514 - loss: 0.1579 - val_accuracy: 0.9732 - val_loss: 0.1007 - learning_rate: 0.0010
Epoch 45/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9556 - loss: 0.1423 - val_accuracy: 0.9737 - val_loss: 0.1011 - learning_rate: 0.0010
Epoch 46/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9515 - loss: 0.1604 - val_accuracy: 0.9760 - val_loss: 0.1013 - learning_rate: 0.0010
Epoch 47/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9537 - loss: 0.1541 - val_accuracy: 0.9740 - val_loss: 0.0996 - learning_rate: 0.0010
Epoch 48/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9516 - loss: 0.1553 - val_accuracy: 0.9751 - val_loss: 0.0980 - learning_rate: 0.0010
Epoch 49/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9548 - loss: 0.1465 - val_accuracy: 0.9739 - val_loss: 0.1027 - learning_rate: 0.0010
Epoch 50/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9560 - loss: 0.1471 - val_accuracy: 0.9766 - val_loss: 0.0972 - learning_rate: 0.0010

# Here, continue like 51, 52...
Epoch 1/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9576 - loss: 0.1388 - val_accuracy: 0.9740 - val_loss: 0.1044 - learning_rate: 0.0010
Epoch 2/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9555 - loss: 0.1413 - val_accuracy: 0.9752 - val_loss: 0.0974 - learning_rate: 0.0010
Epoch 3/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9603 - loss: 0.1275 - val_accuracy: 0.9759 - val_loss: 0.1046 - learning_rate: 0.0010
Epoch 4/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9576 - loss: 0.1362 - val_accuracy: 0.9728 - val_loss: 0.1090 - learning_rate: 0.0010
Epoch 5/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9631 - loss: 0.1195 - val_accuracy: 0.9726 - val_loss: 0.1110 - learning_rate: 0.0010
Epoch 6/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9622 - loss: 0.1257 - val_accuracy: 0.9751 - val_loss: 0.1033 - learning_rate: 0.0010
Epoch 7/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9604 - loss: 0.1284 - val_accuracy: 0.9752 - val_loss: 0.1027 - learning_rate: 0.0010
Epoch 8/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9609 - loss: 0.1268 - val_accuracy: 0.9743 - val_loss: 0.0997 - learning_rate: 0.0010
Epoch 9/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9597 - loss: 0.1321 - val_accuracy: 0.9748 - val_loss: 0.1016 - learning_rate: 0.0010
Epoch 10/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.9648 - loss: 0.1175
Epoch 10: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9648 - loss: 0.1175 - val_accuracy: 0.9717 - val_loss: 0.1164 - learning_rate: 0.0010
Epoch 11/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9693 - loss: 0.1024 - val_accuracy: 0.9777 - val_loss: 0.1004 - learning_rate: 1.0000e-04
Epoch 12/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9732 - loss: 0.0883 - val_accuracy: 0.9773 - val_loss: 0.0993 - learning_rate: 1.0000e-04
Epoch 13/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9735 - loss: 0.0864 - val_accuracy: 0.9763 - val_loss: 0.1041 - learning_rate: 1.0000e-04
Epoch 14/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 32s 39ms/step - accuracy: 0.9744 - loss: 0.0838 - val_accuracy: 0.9777 - val_loss: 0.1015 - learning_rate: 1.0000e-04
Epoch 15/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9741 - loss: 0.0821 - val_accuracy: 0.9779 - val_loss: 0.1035 - learning_rate: 1.0000e-04
Epoch 16/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9742 - loss: 0.0825 - val_accuracy: 0.9779 - val_loss: 0.1026 - learning_rate: 1.0000e-04
Epoch 17/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9759 - loss: 0.0796 - val_accuracy: 0.9771 - val_loss: 0.1034 - learning_rate: 1.0000e-04
Epoch 18/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step - accuracy: 0.9776 - loss: 0.0727
Epoch 18: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9776 - loss: 0.0727 - val_accuracy: 0.9768 - val_loss: 0.1024 - learning_rate: 1.0000e-04
Epoch 19/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9754 - loss: 0.0806 - val_accuracy: 0.9782 - val_loss: 0.1013 - learning_rate: 1.0000e-05
Epoch 20/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 38ms/step - accuracy: 0.9763 - loss: 0.0754 - val_accuracy: 0.9785 - val_loss: 0.1013 - learning_rate: 1.0000e-05
Epoch 21/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9756 - loss: 0.0767 - val_accuracy: 0.9783 - val_loss: 0.1015 - learning_rate: 1.0000e-05
Epoch 22/50
810/810 ━━━━━━━━━━━━━━━━━━━━ 31s 39ms/step - accuracy: 0.9772 - loss: 0.0740 - val_accuracy: 0.9783 - val_loss: 0.1017 - learning_rate: 1.0000e-05
Epoch 22: early stopping
"""

if __name__ == "__main__":
  data = parse_training_logs(logs)
  plot_training_curves(data)