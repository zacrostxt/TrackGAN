import tensorflow as tf
import numpy as np




class gan_metric:
  def __init__(self, GAN_module) -> None:
    # To distinguish between discriminator and generator
    self.GAN_module = GAN_module
    

# A generic metric wrapper for a discriminator network in GANs
class gan_discriminator_metric(gan_metric):

  def __init__(self, metric, name, GAN_module, **kwargs) -> None:
    # Init parent
    super().__init__(GAN_module)

    # Create a new instance of the metric for each input flow
    self.real_input_metric = metric( **kwargs)
    self.fake_input_metric = metric( **kwargs)
    self.name = name
    # Save history
    self.real_metric_data = []
    self.fake_metric_data = []
  

  def compute_metric(self, output_on_real_images, output_on_fake_images):
    # For smooting instead of tf.ones_like(real_output) , just put 0.9
    _ = self.real_input_metric.update_state(tf.ones_like(output_on_real_images), output_on_real_images)
    _ = self.fake_input_metric.update_state(tf.zeros_like(output_on_fake_images), output_on_fake_images)
  
  # During Training reset the state every epoch/step
  def reset_state(self):
    # Store data before resetting
    self.real_metric_data.append( self.real_input_metric.result())
    self.fake_metric_data.append( self.fake_input_metric.result())

    # Reset
    self.real_input_metric.reset_state()
    self.fake_input_metric.reset_state()
  
  # Return the history so far
  def get_history(self):
    return self.real_metric_data, self.fake_metric_data


  def display_results(self ):
    # = self.real_input_metric.result().numpy()
    real_result = self.real_input_metric.result()
    fake_result = self.fake_input_metric.result()

    #print( f"Accuracy on Real : {disc_acc_on_real_image} \n Accuracy on Fake : {disc_acc_on_fake_image} " )
    tf.print( f"{self.name} on Real {real_result:.4} \n{self.name} on Fake {fake_result:.4}")











  