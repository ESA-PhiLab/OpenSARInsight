from l0_to_range_compressed import compress, compress_multiple, compress_split
from l0_to_range_compressed.utilities.paths import test_compression_dict, test_matching_list
from time import perf_counter

def range_compression_demo():
    """
    Demonstrates the range compression process using test parameters.
    """
    tcd = test_compression_dict
    print("1. Executing single compression test.")
    start_time = perf_counter()
    compressed_img_fp = compress(
	    patch_path=tcd['patch_path'], 
        header_path=tcd['header_path'], 
        cal_data_path=tcd['cal_data_path'], 
        cal_header_path=tcd['cal_header_path'], 
        rescaled_output_dir=tcd['rescaled_output_dir']
    )
    end_time = perf_counter()
    print(f"Compression completed in {end_time - start_time:.2f} seconds")
    print(f"Compressed image saved at: {compressed_img_fp}")

    rescaled_output_path = tcd.pop('rescaled_output_dir')
    print("2. Executing multiple compression test.")
    start_time = perf_counter()
    results_path = compress_multiple(
        matches=test_matching_list,
        rescaled_output_dir=rescaled_output_path,
    )
    end_time = perf_counter()
    print(f"Multiple compression completed in {end_time - start_time:.2f} seconds")
    print(f"Compressed images saved at: {results_path}")

    print("3. Executing batch processing test.")    
    start_time = perf_counter()
    # limit patches to 4 for demo purposes
    batch_results = compress_split(
        split='test',
        dataset='vessel',
        rescaled_output_dir=rescaled_output_path,
        limit=4 
    )
    end_time = perf_counter()
    print(f"Batch compression completed in {end_time - start_time:.2f} seconds")
    print(f"Batch compression results: {batch_results}")

if __name__ == "__main__":
    range_compression_demo()
