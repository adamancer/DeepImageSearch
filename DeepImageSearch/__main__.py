import argparse
import os
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from DeepImageSearch import Load_Data, Search_Setup


class ParserWithError(argparse.ArgumentParser):
    """Modifies parser to return help if command contains an error"""

    def error(self, message):
        """Show help text if command contains an error

        From http://stackoverflow.com/questions/4042452

        Parameters
        ----------
        message : str
           the message to write to stderr
        """
        sys.stderr.write(f"error: {message}\n")
        self.print_help()
        sys.exit(2)


def main(args=None):

    parser = ParserWithError()
    subparsers = parser.add_subparsers(dest="command")
    search = subparsers.add_parser("search", description="search for an image")
    update = subparsers.add_parser("update", description="update reference images")

    search.add_argument("--numresults", type=int, default=15, help="number of images to return")
    search.add_argument("--model", default="resnet152", help="identifier for the model")

    update.add_argument("paths", nargs="+", help="paths containing images to add")
    update.add_argument("--limit", type=int, help="number of images to add")
    update.add_argument("--model", default="resnet152", help="identifier for the model")

    args = parser.parse_args()

    if args.command == "search":

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        # Load existing model
        st = Search_Setup(image_list=None, model_name=args.model, pretrained=True)
        st.get_image_metadata_file()
        
        while True:
            
            root = Tk()
            path = askopenfilename(parent=root, title='Choose a file')
            root.destroy()
            
            print(f"Query:   {path}")
            st.plot_similar_images(image_path=path, number_of_images=args.numresults)
            print("Results:")
            results = st.get_similar_images(path, number_of_images=args.numresults)
            for i, path in enumerate(results.values()):
                print(f" {i + 1}. {path}")
            resp = input("Press ENTER to run another search or 'q` to exit: ")
            if resp.lower() == "q":
                break

    elif args.command == "update":

        limit = None if not hasattr(update, "limit") else update.limit

        # Load reference images
        images = Load_Data().from_folder(args.paths)

        # Load model
        try:
            open(os.path.join("metadata-files", args.model, "image_data_features.pkl"))
        except FileNotFoundError:
            st = Search_Setup(image_list=images[:1], model_name=args.model, pretrained=True)
            st.run_index()
            if limit is not None:
                limit -= 1
        else:
            st = Search_Setup(image_list=None, model_name=args.model, pretrained=True)

        # Add reference images missing from the index
        df = st.get_image_metadata_file()
        paths = set(df["images_paths"])
        new_images = [p for p in images if p not in paths]
        if limit:
            new_images = new_images[:limit]
        if new_images:
            st.add_images_to_index(new_images)
        print(f"{len(st.get_image_metadata_file()):,} images in index")

    else:
        raise ValueError(f"Must specify a subcommand: {['search', 'update']}")


if __name__ == "__main__":
    main()