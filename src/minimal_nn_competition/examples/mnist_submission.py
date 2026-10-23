"""Train a small MNIST classifier and write a submission."""

from functools import partial

from minimal_nn_competition.examples.classifier_submission import build_model, run, train_classifier

create_model = build_model
train = partial(train_classifier, benchmark="mnist")


def main(argv=None):
    return run("MNIST", "data/mnist-train.pt", "models/mnist/example", argv)


if __name__ == "__main__":
    raise SystemExit(main())
