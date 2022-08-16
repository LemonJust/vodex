from sqlite3 import connect


# TODO : make load static method

class DbWriter:
    """
    Writes information to the database.
    Database interface that abstracts the SQLite calls.

    Code adopted from https://www.devdungeon.com/content/python-sqlite3-tutorial
    Maybe better stuff here : https://www.sqlitetutorial.net/sqlite-python/creating-tables/

    Some thoughts on try-catch :
    https://softwareengineering.stackexchange.com/questions/64180/good-use-of-try-catch-blocks
    """

    def __init__(self, connection):
        self.connection = connection
        self.connection.execute("PRAGMA foreign_keys = 1")

    def save(self, file_name):
        """
        Backup a database to a file.
        Will CLOSE connection to the database in memory!
        file_name : "databasename.db"
        """

        def progress(status, remaining, total):
            print(f'Copied {total - remaining} of {total} pages...')

        backup_db = connect(file_name)
        with backup_db:
            self.connection.backup(backup_db, progress=progress)
        self.connection.close()
        backup_db.close()

    @classmethod
    def create(cls):
        """
        Creates an empty DB to the experiment in memory
        """
        # For an in-memory only database:
        memory_db = connect(':memory:')
        return cls(memory_db)

    @classmethod
    def load(cls, file_name):
        """
        Load the contents of a database file on disk to a
        transient copy in memory without modifying the file
        """
        disk_db = connect(file_name)
        memory_db = connect(':memory:')
        disk_db.backup(memory_db)
        disk_db.close()
        # Now use `memory_db` without modifying disk db
        return cls(memory_db)

    def populate(self, volumes=None, annotations=None):
        """
        Creates the tables if they don't exist and fills the provided data.

        :param volumes: mapping of frames to volumes, and to slices in volumes, frames per volume
        :type volumes: VolumeManager
        :param annotations: mapping of frames to labels, list of annotations
        :type annotations: [Annotation]
        :return: None
        """
        # TODO : add warnings when you are trying to write the same data again

        # will only create if they don't exist
        self._create_tables()
        # TODO : write cases for files and frames not None

        if volumes is not None:
            self._populate_Options(volumes.file_manager, volumes)
            self._populate_Files(volumes.file_manager)
            self._populate_Frames(volumes.frame_manager)
            self._populate_Volumes(volumes)

        if annotations is not None:
            for annotation in annotations:
                self._populate_AnnotationTypes(annotation)
                self._populate_AnnotationTypeLabels(annotation)
                self._populate_Annotations(annotation)
                if annotation.cycle is not None:
                    self._populate_Cycles(annotation)
                    self._populate_CycleIterations(annotation)

    def get_n_frames(self):
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM Frames")
            n_frames = cursor.fetchone()[0]
        except Exception as e:
            print(f"Could not get total number of frames from Frames because {e}")
            raise e
        finally:
            cursor.close()
        return n_frames

    def _create_tables(self):

        # TODO : change UNIQUE(a, b)
        # into primary key over both columns a and b where appropriate

        db_cursor = self.connection.cursor()

        sql_create_Options_table = """
            CREATE TABLE IF NOT EXISTS "Options" (
            "Key"	TEXT NOT NULL UNIQUE,
            "Value"	TEXT NOT NULL,
            "Description"	TEXT,
            PRIMARY KEY("Key")
            )
            """
        db_cursor.execute(sql_create_Options_table)

        sql_create_Files_table = """
            CREATE TABLE IF NOT EXISTS "Files" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "FileName"	TEXT NOT NULL UNIQUE,
            "NumFrames"	INTEGER NOT NULL,
            PRIMARY KEY("Id" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_Files_table)

        sql_create_AnnotationTypes_table = """
            CREATE TABLE IF NOT EXISTS "AnnotationTypes" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "Name"	TEXT NOT NULL UNIQUE,
            "Description"	TEXT,
            PRIMARY KEY("Id" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_AnnotationTypes_table)

        sql_create_Frames_table = """
            CREATE TABLE IF NOT EXISTS "Frames" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "FrameInFile"	INTEGER NOT NULL,
            "FileId"	INTEGER NOT NULL,
            PRIMARY KEY("Id" AUTOINCREMENT),
            FOREIGN KEY("FileId") REFERENCES "Files"("Id")
            )
            """
        db_cursor.execute(sql_create_Frames_table)

        sql_create_Cycles_table = """
            CREATE TABLE IF NOT EXISTS "Cycles" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "AnnotationTypeId"	INTEGER NOT NULL UNIQUE,
            "Structure"	TEXT NOT NULL,
            FOREIGN KEY("AnnotationTypeId") REFERENCES "AnnotationTypes"("Id"),
            PRIMARY KEY("Id" AUTOINCREMENT)
            )
            """
        db_cursor.execute(sql_create_Cycles_table)

        sql_create_AnnotationTypeLabels_table = """
            CREATE TABLE IF NOT EXISTS "AnnotationTypeLabels" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "AnnotationTypeId"	INTEGER NOT NULL,
            "Name"	TEXT NOT NULL,
            "Description"	TEXT,
            PRIMARY KEY("Id" AUTOINCREMENT),
            FOREIGN KEY("AnnotationTypeId") REFERENCES "AnnotationTypes"("Id"),
            UNIQUE("AnnotationTypeId","Name")
            )
            """
        db_cursor.execute(sql_create_AnnotationTypeLabels_table)

        sql_create_Annotations_table = """
            CREATE TABLE IF NOT EXISTS "Annotations" (
            "FrameId"	INTEGER NOT NULL,
            "AnnotationTypeLabelId"	INTEGER NOT NULL,
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id"),
            FOREIGN KEY("AnnotationTypeLabelId") REFERENCES "AnnotationTypeLabels"("Id"),
            UNIQUE("FrameId","AnnotationTypeLabelId")
            )
            """
        db_cursor.execute(sql_create_Annotations_table)

        sql_create_CycleIterations_table = """
            CREATE TABLE IF NOT EXISTS "CycleIterations" (
            "FrameId"	INTEGER NOT NULL,
            "CycleId"	INTEGER NOT NULL,
            "CycleIteration"	INTEGER NOT NULL,
            FOREIGN KEY("CycleId") REFERENCES "Cycles"("Id"),
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id")
            )
            """
        db_cursor.execute(sql_create_CycleIterations_table)

        sql_create_Volumes_table = """
            CREATE TABLE IF NOT EXISTS "Volumes" (
            "FrameId"	INTEGER NOT NULL UNIQUE,
            "VolumeId"	INTEGER NOT NULL,
            "SliceInVolume"	INTEGER NOT NULL,
            PRIMARY KEY("FrameId" AUTOINCREMENT),
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id")
            UNIQUE("VolumeId","SliceInVolume")
            )
            """
        db_cursor.execute(sql_create_Volumes_table)

        db_cursor.close()

    def _populate_Options(self, file_manager, volume_manager):
        """
        Options: dictionary with key - value pairs
        another way of dealing with Errors : ( more pretty ??? )
        https://www.w3resource.com/python-exercises/sqlite/python-sqlite-exercise-6.php
        """
        row_data = [("data_dir", file_manager.data_dir.as_posix()),
                    ("frames_per_volume", volume_manager.fpv),
                    ("num_head_frames", volume_manager.n_head),
                    ("num_tail_frames", volume_manager.n_tail),
                    ("num_full_volumes", volume_manager.full_volumes)]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Options (Key, Value) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Options because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Files(self, file_manager):
        """
        file_name : list with filenames per file (str)
        num_frames : list with number of frames per file (int)
        """
        row_data = [(filename, frames) for
                    filename, frames in zip(file_manager.file_names, file_manager.num_frames)]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Files (FileName, NumFrames) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Files because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Frames(self, frame_manager):
        """
        Something like :
        insert into tab2 (id_customers, value)
        values ((select id from tab1 where customers='john'), 'alfa');
        but in SQLite
        https://www.tutorialspoint.com/sqlite/sqlite_insert_query.htm
        """
        # adding +1 since the frame_to_file is indexing files from 0 and sqlite gave files IDs from 1
        row_data = [(frame_in_file, frame_to_file + 1) for
                    frame_in_file, frame_to_file in zip(frame_manager.frame_in_file, frame_manager.frame_to_file)]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Frames (FrameInFile, FileId) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Frames because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Volumes(self, volume_manager):

        row_data = [(volume_id, slice_in_volume) for
                    volume_id, slice_in_volume in zip(volume_manager.frame_to_vol, volume_manager.frame_to_z)]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Volumes (VolumeId, SliceInVolume) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Volumes because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_AnnotationTypes(self, annotation):
        """
        """
        row_data = (annotation.name, annotation.info)
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO AnnotationTypes (Name, Description) VALUES (?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to AnnotationTypes because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_AnnotationTypeLabels(self, annotation):

        row_data = [(label.group, label.name, label.description)
                    for label in annotation.labels.states]

        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO AnnotationTypeLabels (AnnotationTypeId, Name, Description) " +
                "VALUES((SELECT Id FROM AnnotationTypes WHERE Name = ?), ?, ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to AnnotationTypeLabels because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Annotations(self, annotation):
        n_frames = self.get_n_frames()
        assert n_frames == annotation.n_frames, f"Number of frames in the annotation, {annotation.n_frames}," \
                                                f"doesn't match" \
                                                f" the expected number of frames {n_frames}"
        frames = range(n_frames)
        row_data = [(frame + 1, label.name, label.group)
                    for frame, label in zip(frames, annotation.frame_to_label)]
        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO Annotations (FrameId, AnnotationTypeLabelId) " +
                "VALUES(?, (SELECT Id FROM AnnotationTypeLabels "
                "WHERE Name = ? " +
                "AND AnnotationTypeId = (SELECT Id FROM AnnotationTypes " +
                "WHERE Name = ?)))",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write group {annotation.name} to Annotations because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_Cycles(self, annotation):
        """
        """
        row_data = (annotation.name, annotation.cycle.to_json())
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO Cycles (AnnotationTypeId, Structure) " +
                "VALUES((SELECT Id FROM AnnotationTypes WHERE Name = ?), ?)",
                row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write to Cycles because {e}")
            raise e
        finally:
            cursor.close()

    def _populate_CycleIterations(self, annotation):
        n_frames = self.get_n_frames()
        assert n_frames == annotation.n_frames, f"Number of frames in the annotation, {annotation.n_frames}," \
                                                f"doesn't match" \
                                                f" the expected number of frames {n_frames}"
        # get cycle id by annotation type
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT Id FROM Cycles " +
            "WHERE AnnotationTypeId = (SELECT Id FROM AnnotationTypes " +
            "WHERE Name = ?)", (annotation.name,))
        cycle_id = cursor.fetchone()[0]
        cursor.close()

        # prepare rows
        frames = range(n_frames)
        row_data = [(frame + 1, cycle_id, iteration)
                    for frame, iteration in zip(frames, annotation.frame_to_cycle)]

        # insert into CycleIterations
        cursor = self.connection.cursor()
        try:
            cursor.executemany(
                "INSERT INTO CycleIterations (FrameId, CycleId, CycleIteration) " +
                "VALUES(?, ?,?)", row_data)
            self.connection.commit()
        except Exception as e:
            print(f"Could not write group {annotation.name} to CycleIterations because {e}")
            raise e
        finally:
            cursor.close()


class DbReader:
    """
    Reads information from the database.
    Database interface that abstracts the SQLite calls.
    """

    def __init__(self, connection):
        self.connection = connection
        self.connection.execute("PRAGMA foreign_keys = 1")

    def get_n_frames(self):
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM Frames")
            n_frames = cursor.fetchone()[0]
        except Exception as e:
            print(f"Could not get total number of frames from Frames because {e}")
            raise e
        finally:
            cursor.close()
        return n_frames

    @classmethod
    def load(cls, file_name):
        """
        Load the contents of a database file on disk to a
        transient copy in memory without modifying the file
        """
        disk_db = connect(file_name)
        memory_db = connect(':memory:')
        disk_db.backup(memory_db)
        disk_db.close()
        # Now use `memory_db` without modifying disk db
        return cls(memory_db)

    def choose_full_volumes(self, frames):
        """
        Chooses the frames from specified frames, that also correspond to full volumes.

        The order of the frames is not preserved!
        The result will correspond to frames sorted in increasing order !

        :param frames: a list of frame IDs
        :type frames: [int]
        :param fpv: frames per volume
        :type fpv: int
        :return: frames IDs from frames. corresponding to slices
        :rtype: [int]
        """
        # create list of frame Ids
        frame_ids = tuple(frames)
        n_frames = len(frame_ids)

        # get the volumes
        cursor = self.connection.cursor()
        try:
            # get n_volumes (frames per volume)
            cursor.execute(
                f"""SELECT Value FROM Options 
                    WHERE Key = "frames_per_volume" """
            )
            fpv = int(cursor.fetchone()[0])

            # get ids of full volumes in the provided frames
            cursor.execute(
                f"""SELECT VolumeId FROM
                    (
                    SELECT VolumeId, count(VolumeId) as N
                    FROM Volumes
                    WHERE FrameId IN ({', '.join(['?'] * n_frames)})
                    GROUP BY VolumeId
                    )
                    WHERE N = ?""", frame_ids + (fpv,)
            )
            volume_ids = [volume[0] for volume in cursor.fetchall()]
            n_volumes = len(volume_ids)

            # get all frames from frames that correspond to full volumes
            cursor.execute(
                f"""SELECT FrameId FROM Volumes
                    WHERE FrameId IN ({', '.join(['?'] * n_frames)})
                    AND VolumeId IN ({', '.join(['?'] * n_volumes)})""", frame_ids + tuple(volume_ids)
            )
            frame_ids = [frame[0] for frame in cursor.fetchall()]
        except Exception as e:
            print(f"Could not choose_full_volumes because {e}")
            raise e
        finally:
            cursor.close()

        return volume_ids, frame_ids

    def choose_frames_per_slices(self, frames, slices):
        """
        Chooses the frames from specified frames, that also correspond to the same slices (continuously)
        in different volumes.
        For example, if slices = [2,3,4] it will choose such frames from "given frames" that also correspond
        to a chunk from slice 2 to slice 4 in all the volumes.
        If there is a frame that corresponds to a slice "2" in a volume,
        but no frames corresponding to the slices "3" and "4" in the SAME volume, such frame will not be picked.

        The order of the frames is not preserved!
        The result will correspond to frames sorted in increasing order !

        :param frames: a list of frame IDs
        :type frames: [int]
        :param slices: a list of slice IDs, order will not be preserved: will be sorted in increasing order
        :type slices: [int]
        :return: frames IDs from frames. corresponding to slices
        :rtype: [int]
        """
        # create list of frame Ids
        frame_ids = tuple(frames)
        slice_ids = tuple(slices)
        n_frames = len(frame_ids)
        n_slices = len(slices)

        # get the volumes
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""SELECT FrameId FROM Volumes
                    WHERE FrameId in ({', '.join(['?'] * n_frames)})
                    AND SliceInVolume IN ({', '.join(['?'] * n_slices)})
                    AND VolumeId IN
                        (
                        SELECT VolumeId FROM
                            (
                            SELECT VolumeId, count(VolumeId) as N 
                            FROM Volumes 
                            WHERE FrameId IN ({', '.join(['?'] * n_frames)})
                            AND SliceInVolume IN ({', '.join(['?'] * n_slices)})
                            GROUP BY VolumeId
                            )
                        WHERE N = ?
                        )""", frame_ids + slice_ids + frame_ids + slice_ids + (n_slices,)
            )

            frame_ids = cursor.fetchall()
        except Exception as e:
            print(f"Could not choose_frames_per_slices because {e}")
            raise e
        finally:
            cursor.close()
        frame_ids = [frame[0] for frame in frame_ids]
        return frame_ids

    def prepare_frames_for_loading(self, frames):
        """
        Finds all the information needed
        1) to load the frames
        2) and shape them back into volumes/slices.
        For each frame returns the image file,
        frame location on the image and corresponding volume.

        The order is not preserved!
        the volume correspond to sorted volumes in increasing order !

        :param frames: a list of frame IDs
        :type frames: [int]
        :return: three lists, the length of the frames :
                data directory, file names,
                image files, frame location on the image, volumes.
        :rtype: str, [str], [str],[int],[int]
        # TODO : break into functions that get the file locations and the stuff relevant to the frames
        # TODO : make one " prepare volumes for loading?
        """
        n_frames = len(frames)

        # get the frames
        cursor = self.connection.cursor()
        try:
            # get data directory
            cursor.execute(
                f"""SELECT Value
                    FROM Options
                    WHERE Key = "data_dir" """)
            data_dir = cursor.fetchone()[0]

            # get file_names
            cursor.execute(
                f"""SELECT FileName
                    FROM Files """)
            file_names = cursor.fetchall()

            # get info for every frame
            cursor.execute(
                f"""SELECT 
                        FileId,
                        FrameInFile,
                        VolumeId
                    FROM 
                        Frames
                        INNER JOIN Volumes 
                        ON Frames.Id = Volumes.FrameId
                    WHERE FrameId IN ({', '.join(['?'] * n_frames)})""", tuple(frames))
            frame_info = cursor.fetchall()

        except Exception as e:
            print(f"Could not prepare_frames_for_loading because {e}")
            raise e
        finally:
            cursor.close()

        file_names = [row[0] for row in file_names]

        file_ids = [row[0] for row in frame_info]
        frame_in_file = [row[1] for row in frame_info]
        volumes = [row[2] for row in frame_info]

        return data_dir, file_names, file_ids, frame_in_file, volumes

    def get_frames_per_volumes(self, volume_ids):
        """
        Finds all the frames that correspond to the specified volumes.
        The order is not preserved!
        the volume correspond to sorted volumes in increasing order !

        :param volume_ids: a list of volume IDs
        :type volume_ids: [int]
        :return: frame IDs
        :rtype: [int]
        """
        ids = volume_ids

        # get the frames
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT FrameId FROM Volumes 
                    WHERE VolumeId IN ({', '.join(['?'] * len(ids))})""", tuple(ids))
            frame_ids = cursor.fetchall()
        except Exception as e:
            print(f"Could not _get_FrameId_from_Volumes because {e}")
            raise e
        finally:
            cursor.close()
        # TODO : get rid of list of tuples?
        #  https://www.reddit.com/r/Python/comments/2iiqri/quick_question_why_are_sqlite_fields_returned_as/
        frame_ids = [frame[0] for frame in frame_ids]
        return frame_ids

    def get_and_frames_per_annotations(self, conditions):
        """
        Chooses the frames that correspond to the specified conditions on annotation. Using "and" logic. Example : if
        you ask for frames corresponding to condition 1 and condition 2 , it will return such frames that both
        condition 1 and condition 2 are True AT THE SAME TIME

        :param conditions: a list of conditions on the annotation labels
        in a form [(group, name),(group, name), ...] where group is a string for the annotation type
        and name is the name of the label of that annotation type. For example [('light', 'on'), ('shape','c')]
        :type conditions: [tuple]
        :return: list of frame Ids that satisfy all the conditions, if there are no such frames, an empty list
        :rtype: list
        """
        # create list of label Ids
        labels_ids = []
        for label_info in conditions:
            labels_ids.append(self._get_AnnotationLabelId(label_info))
        labels_ids = tuple(labels_ids)
        n_labels = len(labels_ids)

        # get the frames
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT FrameId FROM 
                    (SELECT FrameId, count(FrameId) as N 
                    FROM Annotations 
                    WHERE AnnotationTypeLabelId IN ({', '.join(['?'] * n_labels)})
                    GROUP BY FrameId)
                    WHERE N = {n_labels}""", labels_ids)
            frame_ids = cursor.fetchall()
        except Exception as e:
            print(f"Could not _get_and_FrameId_from_Annotations because {e}")
            raise e
        finally:
            cursor.close()
        # TODO : get rid of list of tuples?
        #  https://www.reddit.com/r/Python/comments/2iiqri/quick_question_why_are_sqlite_fields_returned_as/
        frame_ids = [frame[0] for frame in frame_ids]
        return frame_ids

    def get_or_frames_per_annotations(self, conditions):
        """
        Chooses the frames that correspond to the specified conditions on annotation. Using "or" logic. Example : if
        you ask for frames corresponding to condition 1 and condition 2 , it will return such frames that either
        condition 1 is true OR condition 2 is True OR both are true.

        :param conditions: a list of conditions on the annotation labels
        in a form [(group, name),(group, name), ...] where group is a string for the annotation type
        and name is the name of the label of that annotation type. For example [('light', 'on'), ('shape','c')]
        :type conditions: [tuple]
        :return:
        :rtype:
        """
        # create list of label Ids
        labels_ids = []
        for label_info in conditions:
            labels_ids.append(self._get_AnnotationLabelId(label_info))
        labels_ids = tuple(labels_ids)
        n_labels = len(labels_ids)

        # get the frames
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT FrameId FROM Annotations 
                    WHERE AnnotationTypeLabelId IN ({', '.join(['?'] * n_labels)})
                    GROUP BY FrameId""", labels_ids)
            frame_ids = cursor.fetchall()
        except Exception as e:
            print(f"Could not _get_or_FrameId_from_Annotations because {e}")
            raise e
        finally:
            cursor.close()
        # TODO : get rid of list of tuples?
        #  https://www.reddit.com/r/Python/comments/2iiqri/quick_question_why_are_sqlite_fields_returned_as/
        frame_ids = [frame[0] for frame in frame_ids]
        return frame_ids

    def _get_AnnotationLabelId(self, label_info):
        """
        Returns the AnnotationLabels.Id for a label , searching by its name and group name.
        :param label_condition: (group, name), where group is the AnnotationType.Name
                            and name is AnnotationTypeLabels.Name
        :type label_condition: tuple
        :return: AnnotationLabels.Id
        :rtype: int
        """
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT Id FROM AnnotationTypeLabels 
                    WHERE AnnotationTypeId = (SELECT Id from AnnotationTypes WHERE Name = ?)
                    and Name = ?""", label_info)
            label_id = cursor.fetchone()
            assert label_id is not None, f"Could not find a label from group {label_info[0]} " \
                                         f"with name {label_info[1]}. " \
                                         f"Are you sure it's been added into the database? "
        except Exception as e:
            print(f"Could not _get_AnnotationLabelId because {e}")
            raise e
        finally:
            cursor.close()
        return label_id[0]

    def _get_SliceInVolume_from_Volumes(self, frames):
        """
        Chooses the slices that correspond to the specified frames.
        The order of the frames is not preserved!
        the volume correspond to frames sorted in increasing order !

        :param frames: a list of frame IDs
        :type frames: [int]
        :return: volume IDs
        :rtype: [int]
        """
        # create list of frame Ids
        frame_ids = tuple(frames)
        n_frames = len(frame_ids)

        # get the volumes
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT SliceInVolume FROM Volumes 
                    WHERE FrameId IN ({', '.join(['?'] * n_frames)})""", frame_ids)
            slice_ids = cursor.fetchall()
        except Exception as e:
            print(f"Could not _get_SliceInVolume_from_Volumes because {e}")
            raise e
        finally:
            cursor.close()
        slice_ids = [slice_id[0] for slice_id in slice_ids]
        return slice_ids

    def _get_VolumeId_from_Volumes(self, frames):
        """
        Chooses the volumes that correspond to the specified frames.
        The order is not preserved!
        the volume correspond to sorted frames in increasing order !

        :param frames: a list of frame IDs
        :type frames: [int]
        :return: volume IDs
        :rtype: [int]
        """
        # create list of frame Ids
        frame_ids = tuple(frames)
        n_frames = len(frame_ids)

        # get the volumes
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT VolumeId FROM Volumes 
                    WHERE FrameId IN ({', '.join(['?'] * n_frames)})""", frame_ids)
            volume_ids = cursor.fetchall()
        except Exception as e:
            print(f"Could not _get_VolumeId_from_Volumes because {e}")
            raise e
        finally:
            cursor.close()
        volume_ids = [volume[0] for volume in volume_ids]
        return volume_ids

# class DbExporter:
#     """
#     Transforms the information from the database into other human-readable formats.
#     """
#
#     def __init__(self, db):
#         self.connection = db
#         db.execute("PRAGMA foreign_keys = 1")
#
#     @classmethod
#     def load(cls, file_name):
#         """
#         Load the contents of a database file on disk to a
#         transient copy in memory without modifying the file
#         """
#         disk_db = connect(file_name)
#         memory_db = connect(':memory:')
#         disk_db.backup(memory_db)
#         disk_db.close()
#         # Now use `memory_db` without modifying disk db
#         return cls(memory_db)
#
#     def list_tables(self):
#         """
#         Shows all the tables in the database
#         """
#         cursor = self.connection.cursor()
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#         print(cursor.fetchall())
#         cursor.close()
#
#     def table_as_df(self, table_name):
#         """
#         Returns the whole table as a dataframe
#         table_name : name of the table you want to see
#         """
#
#         cursor = self.connection.cursor()
#         try:
#             query = cursor.execute(f"SELECT * From {table_name}")
#             cols = [column[0] for column in query.description]
#             df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
#         except Exception as e:
#             print(f"Could not show {table_name} because {e}")
#             raise e
#         finally:
#             cursor.close()
#         return df