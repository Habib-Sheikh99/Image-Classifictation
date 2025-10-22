import matplotlib.pyplot as plt
import seaborn as sns
class Visualization : 
        def plot_comparison(self, data1, data2, color1='green', color2='orange', label1='Actual', label2='Predicted'):
                    sns.set()
                    plt.figure(figsize=(7,7), facecolor='grey')
                    plt.plot(data1, color=color1, label=label1)
                    plt.plot(data2, color=color2, label=label2)
                    plt.legend()
                    plt.show()
        def plot_history(self, loss, val_loss):
                    sns.set()
                    plt.figure(figsize=(5,5), facecolor='grey')
                    plt.plot(loss, color='orange', label="Training loss")
                    plt.plot(val_loss, color='purple', label="Testing loss")
                    plt.legend()
                    plt.show()

class Neaural_Network:
    pass
