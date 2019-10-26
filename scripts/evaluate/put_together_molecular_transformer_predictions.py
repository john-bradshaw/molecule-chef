"""
Put together Molecular Transformer predictions

Usage:
  put_together_molecular_transformer_predictions.py <tokenized_products_path> <output_path> [--nbest=<nbest>]

Options:
  --nbest=<nbest>  The number of predictions the transformer has made for each reactant set [default: 1].

"""

from docopt import docopt


def main():
    arguments = docopt(__doc__)

    with open(arguments['<tokenized_products_path>'], 'r') as fo:
        lines = [l.strip() for l in fo.readlines()]

    n_best = int(arguments['--nbest'])

    top_1 = []
    for i, products in enumerate(lines):
        product_put_together = ''.join(products.split())
        if i % n_best == 0:
            top_1.append(product_put_together)

    with open(arguments['<output_path>'], 'w') as fo:
        fo.writelines('\n'.join(top_1))


if __name__ == '__main__':
    main()
