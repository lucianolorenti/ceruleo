from  tensorflow.keras.callbacks import Callback

class PredictionCallback(Callback):    
    def on_epoch_end(self, epoch, logs={}):
        
        y_pred = model.predict(validation_dataset)
        y_true = model.true_values(validation_dataset)
        fig, ax = plot_true_vs_predicted(y_true, y_pred, figsize=(17, 5), ylabel='Seconds [s]')
        ax.plot(pd.Series(np.squeeze(y_pred)).rolling(100).mean(), label='Smoothed')
        ax.legend()
        fig.savefig('/home/luciano/aa1.png', dpi=fig.dpi)
        
        plt.close(fig)