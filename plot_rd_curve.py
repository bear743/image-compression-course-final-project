import matplotlib.pyplot as plt
from image_compression import imageEncoder, imageDecoder

def plot_rd_curve(image_path, q_steps=None, save_path='rd_curve.png'):
    """
    Plot Rate-Distortion curve for multiple quantization step sizes
    Args:
        image_path: Path to input image
        q_steps: List of quantization step sizes to test
        save_path: Where to save the plot
    """
    if q_steps is None:
        q_steps = [1, 2, 4, 8, 16, 32, 64]
    
    bitrates = []
    psnrs = []
    
    for q in q_steps:
        print(f"Testing q={q}...")
        # Encode
        bitrate = imageEncoder(image_path, q)
        # Decode
        psnr = imageDecoder('image.bit', q, image_path)
        
        bitrates.append(bitrate)
        psnrs.append(psnr)
        print(f"q={q}: Bitrate={bitrate:.4f} bpp, PSNR={psnr:.2f} dB")
    
    # Plot R-D curve
    plt.figure(figsize=(10, 6))
    plt.plot(bitrates, psnrs, 'o-', linewidth=2)
    
    # Annotate points with q values
    for i, q in enumerate(q_steps):
        plt.annotate(f"q={q}", (bitrates[i], psnrs[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title(f'Rate-Distortion Curve ({image_path})')
    plt.xlabel('Bitrate (bits per pixel)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    # Save and show
    plt.savefig(save_path)
    plt.show()
    return bitrates, psnrs

# Example usage:
if __name__ == "__main__":
    # Test on sample image
    image_path = 'lena.bmp'
    bitrates, psnrs = plot_rd_curve(image_path, save_path=f'{image_path}_rd_curve.png')
    
    # Print results in a table
    print("\nQuantization | Bitrate (bpp) | PSNR (dB)")
    print("----------------------------------------")
    for q, r, d in zip([1, 2, 4, 8, 16, 32, 64], bitrates, psnrs):
        print(f"{q:11.1f} | {r:13.4f} | {d:9.2f}")