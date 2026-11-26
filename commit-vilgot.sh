#!/bin/sh
GIT_AUTHOR_NAME="Person One" \
GIT_AUTHOR_EMAIL="person1@example.com" \
GIT_COMMITTER_NAME="Person One" \
GIT_COMMITTER_EMAIL="person1@example.com" \
git commit "$@"
