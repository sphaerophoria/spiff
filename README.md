# Spiff, a diff tool for me

Install
```
cargo install --path .
```

And add to your gitconfig
```
[difftool "spiff"]
        cmd = spiff_gui $LOCAL $REMOTE
```

And run it
```
git difftool -d HEAD~1..
```

