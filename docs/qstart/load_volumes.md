# Load Volumes

## Load based on volume ID

The volume ID is it's number from the beginning of the recording (**volume IDs start at 0!**).

You can use `list_volumes` method to get a list of all available volume IDs. You will see ID `-1` if there is a non-full volume at the beginning of the recording, and `-2` for the non-full volume at the end of the recording.
```{.py3 .in}
import vodex as vx
experiment = vx.Experiment.load("test.db")
experiment.list_volumes()
```
``` {.text .yaml .no-copy }
array([-2,  0,  1,  2,  3])
```

Here's how you can load some volumes:
```{.py3 .in}
# so we have 4 full volumes : 0,1,2,3 and some extra frames at the end (-2)
# let's load two full volumes , 0 and 2:
vol02 = experiment.load_volumes([0,2])
print(vol02.shape)
```
``` {.text .yaml .no-copy }
(2, 10, 200, 200)
```
We loaded two volumes, each has 10 slices and each frame is 200 by 200 pixels.

If you want to have a look at the fames at the end of the recording that do not form a full volume,
 you can do so by asking for the volume ID `-2`:
```{.py3 .in}
volm2 = experiment.load_volumes([-2])
print(volm2.shape)
```
``` {.text .yaml .no-copy }
((1, 2, 200, 200))
```
Notice how this volume only has 2 frames!

## Load based on experimental conditions

You will use to method `choose_volumes` to get the IDs of the volumes that correspond to certain conditions and then `load_volumes` to actually load the volumes.

You can choose volumes based on any annotation. Let's see what volumes correspond to light "off":
```{.py3 .in}
light_off_ids = experiment.choose_volumes([("light","off")])
light_off_ids
```
```{.text .yaml .no-copy }
[0, 3]
```
Volumes 0 and 3 have light off for every frame. Use `load_volumes` to load these volumes:
```{.py3 .in}
vol_light_off = experiment.load_volumes(light_off_ids)
```

You can also combine the conditions with a logical `and` or logical `or`.
You can go and experiment with this: what volumes correspond to the times when the light is off `or` the c label is c2 ?
```{.py3 .in}
light_off_ids = experiment.choose_volumes([("light","off"),("c label","c2")], logic="or")
light_off_ids
```
```{.text .yaml .no-copy }
[0, 1, 3]
```
How about such volumes that the light is "off" `and` "on"
```{.py3 .in}
light_off_ids = experiment.choose_volumes([("light","off"),("light","on")], logic="and")
light_off_ids
```
```{.text .yaml .no-copy }
[]
```
We got an empty list : no such volumes, since the light can't be on `and` off at the same time. What if we had an `or`?  
```{.py3 .in}
light_off_ids = experiment.choose_volumes([("light","off"),("light","on")], logic="or")
light_off_ids
```
```{.text .yaml .no-copy }
[0, 1, 2, 3]
```
on `or` off returns all full volumes, since for every frame the light was either on or off.
