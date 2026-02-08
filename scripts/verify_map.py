import numpy as np
from PIL import Image
import os
import argparse


def verify_and_fix_map(map_name):
    """
    Verify map image format and fix if needed.
    
    The F1TENTH gym expects:
    - Grayscale PNG (single channel or RGB with same values)
    - Black (0) = occupied space (walls)
    - White (255) = free drivable space
    - Gray (intermediate) = unknown
    """
    
    map_path = os.environ.get('MAP_PATH')
    if map_path is None:
        raise RuntimeError('MAP_PATH not set')
    map_file = map_path+'_map.png'
    backup_file = map_path+'_mapOG.png'

    print(f"Checking map: {map_file}")
    
    # Check if file exists
    if not os.path.exists(map_file):
        print(f"\tMap file not found: {map_file}")
        return False
    
    # Load image
    try:
        img = Image.open(map_file)
        print(f"\tImage loaded successfully")
        print(f"\tMode: {img.mode}")
        print(f"\tSize: {img.size}")
        
        # Convert to numpy array
        img_array = np.array(img)
        print(f"\tArray shape: {img_array.shape}")
        print(f"\tArray dtype: {img_array.dtype}")
        print(f"\tValue range: [{img_array.min()}, {img_array.max()}]")
        
        # Check if it's the right format
        issues = []
        
        # Should be 2D (grayscale) or 3D with last dim = 3 or 4 (RGB/RGBA)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                print("\tImage has alpha channel (RGBA), converting to RGB")
                img_array = img_array[:, :, :3]
                issues.append("alpha_channel")
            
            # Check if all channels are the same (grayscale stored as RGB)
            if np.allclose(img_array[:,:,0], img_array[:,:,1]) and \
               np.allclose(img_array[:,:,1], img_array[:,:,2]):
                print("   ℹ️  Image is RGB but effectively grayscale")
                img_array = img_array[:, :, 0]  # Take first channel
            else:
                print("\tImage has different RGB channels, converting to grayscale")
                img_array = np.mean(img_array, axis=2).astype(np.uint8)
                issues.append("rgb_not_grayscale")
        
        elif len(img_array.shape) != 2:
            print(f"\tUnexpected image dimensions: {img_array.shape}")
            return False
        
        # Check data type
        if img_array.dtype != np.uint8:
            print(f"\tWrong dtype: {img_array.dtype}, converting to uint8")
            img_array = img_array.astype(np.uint8)
            issues.append("wrong_dtype")
        
        # Check value range
        if img_array.max() <= 1:
            print("\tValues in range [0,1], scaling to [0,255]")
            img_array = (img_array * 255).astype(np.uint8)
            issues.append("normalized_values")
        
        # Check for proper occupancy grid values
        unique_vals = np.unique(img_array)
        print(f"\tUnique pixel values: {len(unique_vals)} values")
        if len(unique_vals) <= 10:
            print(f"      {unique_vals}")
        
        # Histogram
        black_pixels = np.sum(img_array == 0)
        white_pixels = np.sum(img_array == 255)
        gray_pixels = img_array.size - black_pixels - white_pixels
        
        if gray_pixels > black_pixels:
            print("\tMore gray pixels than black pixels, check occupancy representation")
            issues.append("too_many_gray_pixels")
            img_array = np.where(img_array < 128, 0, 255).astype(np.uint8)  # Ensure all pixels are either black or white
        
        black_pixels = np.sum(img_array == 0)
        white_pixels = np.sum(img_array == 255)
        gray_pixels = img_array.size - black_pixels - white_pixels
        
        print(f"\tPixel distribution:")
        print(f"\t  Black (0):     {black_pixels:8d} ({100*black_pixels/img_array.size:.1f}%)")
        print(f"\t  White (255):   {white_pixels:8d} ({100*white_pixels/img_array.size:.1f}%)")
        print(f"\t  Gray (other):  {gray_pixels:8d} ({100*gray_pixels/img_array.size:.1f}%)")
        
        # Save fixed version if needed
        if issues:
            # Backup original if not already done
            if not os.path.exists(backup_file):
                img.save(backup_file)
                print(f"\n💾 Backed up original to: {backup_file}")
            
            # Save fixed version
            fixed_img = Image.fromarray(img_array, mode='L')  # 'L' = grayscale
            fixed_img.save(map_file)
            print(f"\tFixed and saved to: {map_file}")
            print(f"\tIssues fixed: {', '.join(issues)}")
        else:
            print(f"\n\tMap image format is correct!")
        
        return True
        
    except Exception as e:
        print(f"\tError loading image: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('map_name', help='Map name')
    args = parser.parse_args()
    verify_and_fix_map(args.map_name)