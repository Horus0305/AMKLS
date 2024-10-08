#pgnmaker
import os

def combine_pgn_files(input_folder, output_file):
    # Open the output file in write mode
    try:
        with open(output_file, 'w', encoding='utf-8') as output_f:
            # Loop through all files in the specified directory
            for filename in os.listdir(input_folder):
                if filename.endswith('.pgn'):  # Check if the file is a PGN file
                    file_path = os.path.join(input_folder, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            # Read the content and write it directly to the output file
                            output_f.write(f.read())
                            output_f.write('\n\n')  # Add extra newlines between files
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

# Specify the input folder and output file
input_folder = r'D:\AMKLS\Lichess Elite Database'  # Update this path
output_file = r'D:\AMKLS\games2400.pgn'  # Update this path

# Combine the PGN files
combine_pgn_files(input_folder, output_file)
