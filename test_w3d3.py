from w3d3_trees import load_data, train_tree, evaluate


def test_load_data_shapes():
    _, X_tr, X_te, y_tr, y_te = load_data()
    assert X_tr.shape[0] == y_tr.shape[0]
    assert X_te.shape[0] == y_te.shape[0]


def test_tree_trains_and_predicts():
    _, X_train, X_test, y_train, y_test = load_data()
    model = train_tree(X_train, y_train, max_depth=3)
    _, test_acc = evaluate(model, X_train, X_test, y_train, y_test)
    assert test_acc > 0.85


def test_unlimited_depth_overfits():
    _, X_train, X_test, y_train, y_test = load_data()
    model = train_tree(X_train, y_train, max_depth=None)
    train_acc, test_acc = evaluate(model, X_train, X_test, y_train, y_test)
    assert train_acc == 1.0
    assert train_acc >= test_acc