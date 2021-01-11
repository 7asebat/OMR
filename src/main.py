import argparse
import os
import OMR

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", help="Input folder path")
parser.add_argument("outputfolder", help="Output folder path")
parser.add_argument("--verbose", help="Show progress", action='store_true')
args = parser.parse_args()

classifiersPath = os.path.join(os.path.dirname(__file__), 'classifiers')

for filename in os.listdir(args.inputfolder):
    inputPath = os.path.join(args.inputfolder, filename)

    basename = filename.split('.')[0]
    if args.verbose:
        print(basename, end='\r\t\t\t')

    if not os.path.isfile(inputPath):
        if args.verbose:
            print(f'ERROR    Not a file')
        continue

    # @note Exceptions are ignored in output format here
    try:
        output = OMR.run_OMR(inputPath, classifiersPath)
        if args.verbose:
            print(f'FINISHED')
    except Exception as e:
        output = ['[]']
        if args.verbose:
            print(f'ERROR    {e}')

    output = ',\n\n'.join(output)
    output = f'{{\n{output}\n}}'
    outputPath = os.path.join(args.outputfolder, f'{basename}.txt')
    with open(outputPath, "w") as text_file:
        text_file.write(output)
