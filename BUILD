package(default_visibility = ["//visibility:public"])

env_args = [
  "--env_type lab",
  "--env_name nav_maze_static_01",
]

py_binary(
    name = "train",
    srcs = ["main.py"],
    args = env_args,
    data = ["//:deepmind_lab.so"],
    main = "main.py"
)

py_binary(
    name = "display",
    srcs = ["display.py"],
    args = env_args,
    data = ["//:deepmind_lab.so"],
    main = "display.py"
)

py_binary(
    name = "visualize",
    srcs = ["visualize.py"],
    args = env_args,
    data = ["//:deepmind_lab.so"],
    main = "visualize.py"
)

py_test(
    name = "test",
    srcs = ["test.py"],
    main = "test.py",
    deps = [":train"],
)

py_binary(
    name = "newtrain",
    srcs = ["newmain.py"],
    args = env_args,
    data = ["//:deepmind_lab.so"],
    main = "newmain.py"
)
