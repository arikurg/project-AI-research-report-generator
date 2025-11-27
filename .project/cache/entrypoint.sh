#!/bin/bash

set -e

# Figure out who we are and how to run commands
# $UID is set by the shell and is the UID of the running user
# $NVWB_UID is set by the ContainerUser configpack during the build
# process and contain the UID of the user that the entrypoint
# should finish executing as.
if [ "$UID" -eq 0 ]
then
    if [ "${NVWB_UID}" -eq 0 ]
    then
        root() { "$@"; }
        user() { "$@"; }
        user_exec() { exec "$@"; }
    else
        root() { "$@"; }
        user() { gosu "${NVWB_UID}:${NVWB_GID}" "$@"; }
        user_exec() { exec gosu "${NVWB_UID}:${NVWB_GID}" "$@"; }
    fi
else
    root() { sudo "$@"; }
    user() { "$@"; }
    user_exec() { exec "$@"; }
fi

# Run any custom entrypoint executable
if [ -n "${NVWB_BASE_ENV_ENTRYPOINT}" ]
then
    if [ ! -x "${NVWB_BASE_ENV_ENTRYPOINT}" ]
    then
        root chmod +x "${NVWB_BASE_ENV_ENTRYPOINT}"
    fi

    echo "Running the Base Environment entrypoint script"
    user "${NVWB_BASE_ENV_ENTRYPOINT}"
fi

# Disable password-less sudo
if [ -f /etc/sudoers.d/workbench ]
then
    root rm /etc/sudoers.d/workbench
    echo "Disabled sudo access"
fi

# Run the command, possibly dropping privileges first
user_exec "$@"

