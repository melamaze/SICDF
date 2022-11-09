import matplotlib.pyplot as plt
from ..config import for_FL as f

class Plot():

    def __init__(self):
        self.accuracy = []
        self.poison_accuracy = []
        self.all_accuracy = []
        self.loss = []


    def draw_plot(self):
        epoch_number = list(range(1, f.epochs + 1))
        
        # accuracy line chart
        plt.plot(epoch_number, self.accuracy, color = 'red')
        plt.xlabel('epoch', fontsize = "10")
        plt.ylabel('accuracy rate', fontsize = "10") 
        plt.xlim(0, f.epochs)
        plt.ylim(0, 1)
        plt.title('Accuracy', fontsize = "16")
        plt.savefig("accuracy.png")
        plt.close() 

        # poison_accuracy line chart
        plt.plot(epoch_number, self.poison_accuracy, color = 'red')
        plt.xlabel('epoch', fontsize = "10")
        plt.ylabel('poison accuracy rate', fontsize = "10") 
        plt.xlim(0, f.epochs)
        plt.ylim(0, 1)
        plt.title('Poison Accuracy', fontsize = "16")
        plt.savefig("poison_accuracy.png")
        plt.close() 

        # all_accuracy line chart
        plt.plot(epoch_number, self.all_accuracy, color = 'red')
        plt.xlabel('epoch', fontsize = "10")
        plt.ylabel('all accuracy rate', fontsize = "10") 
        plt.xlim(0, f.epochs)
        plt.ylim(0, 1)
        plt.title('All Accuracy', fontsize = "16")
        plt.savefig("all_accuracy.png")
        plt.close() 

        # loss line chart
        plt.plot(epoch_number, self.loss, color = 'red')
        plt.xlabel('epoch', fontsize = "10")
        plt.ylabel('loss', fontsize = "10") 
        plt.xlim(0, f.epochs)
        plt.ylim(0, 10)
        plt.title('Loss', fontsize = "16")
        plt.savefig("loss.png")
        plt.close() 



