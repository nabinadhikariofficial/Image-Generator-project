import os
import tensorflow as tf
import matplotlib.pyplot as plt
from config import *
from data_loader import load_and_preprocess_dataset
from trainer import setup_training, train_step, sample

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading dataset...")
    train_ds = load_and_preprocess_dataset()
    train_dataset = (
        train_ds.cache()
        .shuffle(buffer_size=1000)
        .batch(BATCH_SIZE)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    print("Building model and setting up training...")
    unet_model, optimizer, loss_fn = setup_training()
    unet_model.summary()
    
    fixed_noise = tf.random.normal((1, IMG_SIZE, IMG_SIZE, CHANNELS))

    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_dataset.take(STEPS_PER_EPOCH):
            loss = train_step(batch, unet_model, optimizer, loss_fn)
            total_loss += loss

        avg_loss = total_loss / STEPS_PER_EPOCH
        print(f"--- Epoch {epoch+1:03d}/{EPOCHS}, Average Loss: {avg_loss.numpy():.4f} ---")

        # Generate and save a sample image every 10 epochs.
        if (epoch + 1) % 10 == 0:
            print(f"Generating and saving sample for epoch {epoch + 1}...")
            generated_images = sample(unet_model, initial_noise=fixed_noise)
            img = generated_images[0].numpy()
            
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1:03d}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    print("\nTraining Complete!")

if __name__ == "__main__":
    main()