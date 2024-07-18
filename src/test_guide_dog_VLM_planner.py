import sys
import unittest

def test_command_node():
    # Create root node
    root = CommandNode("root")

    # Create child nodes
    child1 = CommandNode("child1")
    child2 = CommandNode("child2")
    child3 = CommandNode("child3")

    # Add child nodes to root
    root.add_child(child1)
    root.add_child(child2)
    root.add_child(child3)

    # Test get_child method
    assert root.get_child("child1") == child1
    assert root.get_child("child2") == child2
    assert root.get_child("child3") == child3
    assert root.get_child("child4") is None

    # Test add_child_to_node method
    child4 = CommandNode("child4")
    child1.add_child_to_node("child1", child4)
    assert child1.get_child("child4") == child4

    # Test remove_child method
    child1.remove_child("child4")
    assert child1.get_child("child4") is None

    # Test create_child method
    child5 = child2.create_child("child5")
    assert child5.command == "child5"
    assert child5.parent == child2

    print("All tests passed!")

test_command_node()