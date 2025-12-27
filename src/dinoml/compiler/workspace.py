import enum


class WorkspaceAllocationMode(enum.Enum):
    eager = 0
    lazy = 1
    fau = 2


def workspace_mode(workspace: WorkspaceAllocationMode) -> str:
    def _impl(workspace):
        if workspace == WorkspaceAllocationMode.eager:
            return "kEager"
        elif workspace == WorkspaceAllocationMode.lazy:
            return "kLazy"
        elif workspace == WorkspaceAllocationMode.fau:
            return "kFau"
        else:
            raise AssertionError(f"unknown workspace {workspace}")

    return f"DinoMLWorkspaceAllocationMode::{_impl(workspace)}"
