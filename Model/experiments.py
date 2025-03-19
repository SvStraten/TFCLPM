import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def experiment(data, learning_object, tags=['Online Continual','Online Continual No Hardbuffer','Online','Online No Hardbuffer']):
    training_losses={}
 
    loss_window_means={}
    update_tags={}
    loss_window_variances={}
    settings={'Online Continual':(True, True),
             'Online Continual No Hardbuffer':(False, True),
             'Online':(True, False),
             'Online No Hardbuffer':(False, False)}
    colors={'Online Continual':'C2',
             'Online Continual No Hardbuffer':'C3',
             'Online':'C4',
             'Online No Hardbuffer':'C5'}

    for tag in tags:
        print("\n{0}".format(tag))
        results=learning_object.method(data,
                                        use_hard_buffer=settings[tag][0],
                                        continual_learning=settings[tag][1])
        future_losses, prediction_results, distribution = results
  
    return future_losses, prediction_results, distribution