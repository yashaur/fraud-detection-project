import numpy as np
from utils.preprocess import load_preprocess
from utils.lgbm import load_model, predict

def precision_recall(y_probs, y_test, thresh: float = 0.5):
   
   y_preds = np.where(y_probs >= thresh, 1, 0)

   y_test = np.array(y_test).reshape(-1)
   
   ap_mask = (y_test == 1) # Obtaining indices of actual positives
   an_mask = (y_test == 0) # Obtaining indices of actual negatives
   pp_mask = (y_preds == 1) # Obtaining indices of predicted postives
   pn_mask = (y_preds == 0) # Obtaining indices of predicted negatives

   tp = np.sum((ap_mask * pp_mask))
   fp = np.sum((an_mask * pp_mask))
   tn = np.sum((an_mask * pn_mask))
   fn = np.sum((ap_mask * pn_mask))

   if __name__ == '__main__':
      output = f'\n\nTrue Positives: {tp}\nFalse Positives: {fp}\nTrue Negatives: {tn}\nFalse Negatives: {fn}\n\n'
      print(output)
         
   precision = 1 if np.isnan(tp / (tp + fp)) else tp / (tp + fp)

   recall = tp / (tp + fn)

   return precision, recall

if __name__ == '__main__':
   
   data_path = 'data/'
   X, y = load_preprocess(data_path)

   model_path = 'model/lgbm.pkl'
   model = load_model(model_path)

   y_probs = predict(model, X)

   precision, recall = precision_recall(y_probs, y, thresh = 1)

   print(f'Precision: {precision*100:.2f}%\nRecall: {recall*100:.2f}%')

