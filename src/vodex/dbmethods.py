"""
This module contains classes that provide a consistent and easy-to-use interface for interacting
with the SQLite database.

DbWriter - A class that writes information to the database. It abstracts the SQLite calls and allows for easy
creation, population, and saving of the database.

DbReader - A class that reads information from the database. It abstracts the SQLite calls and allows for easy
querying and retrieval of data from the database.

DbExporter - A class that transforms the information from the database into the core classes, such as 'FileManager',
'VolumeManager', 'Labels', 'Timeline', 'Cycle' and 'Annotation', making it easier to view and edit with the data.
"""

from sqlite3 import connect, Connection

from .core import *
from .utils import list_of_int

from typing import Union, List


class DbWriter:
    """
    Writes information to the database.
    Database interface that abstracts the SQLite calls.

    Args:
        connection: connection to the SQL database
    """

    def __init__(self, connection: Connection):
        self.connection = connection
        self.connection.execute("PRAGMA foreign_keys = 1")

    def save(self, file_name: str):
        """
        Backup a database to a file.
        Will CLOSE connection to the database in memory!
        Args:
            file_name : the name of the file to save database to, EX.: 'databasename.db'
        """

        def progress(status, remaining, total):
            print(f'Copied {total - remaining} of {total} pages...')

        backup_db = connect(file_name)
        with backup_db:
            self.connection.backup(backup_db, progress=progress)
        # self.connection.close()
        backup_db.close()

    @classmethod
    def create(cls):
        """
        Creates an empty DB for the experiment in memory.
        """
        # For an in-memory only database:
        memory_db = connect(':memory:')
        return cls(memory_db)

    @classmethod
    def load(cls, file_name: str):
        """
        Load the contents of a database file on disk to a
        transient copy in memory without modifying the file.
        """
        disk_db = connect(file_name)
        memory_db = connect(':memory:')
        disk_db.backup(memory_db)
        disk_db.close()
        # Now use `memory_db` without modifying disk db
        return cls(memory_db)

    def populate(self, volumes: VolumeManager = None, annotations: List[Annotation] = None):
        """
        Creates the tables if they don't exist and fills with the provided data.
        Args:
            volumes: mapping of frames to volumes, and to slices in volumes, frames per volume
            annotations: mapping of frames to labels, list of annotations
        """
        # will only create if they don't exist
        self._create_tables()
        # TODO : write cases for files and frames not None

        if volumes is not None:
            self._populate_Options(volumes.file_manager, volumes)
            self._populate_Files(volumes.file_manager)
            self._populate_Frames(volumes.frame_manager)
            self._populate_Volumes(volumes)

        if annotations is not None:
            self.add_annotations(annotations)

    def add_annotations(self, annotations: List[Annotation]):
        """
        Adds a list of annotations to the database.
        Does NOT save the database after adding.
        To keep this change in the future, you need to save the database after adding.
        Args:
            annotations: mapping of frames to labels, list of annotations
        """
        # TODO : add list checking
        for annotation in annotations:
            self._populate_AnnotationTypes(annotation)
            self._populate_AnnotationTypeLabels(annotation)
            self._populate_Annotations(annotation)
            if annotation.cycle is not None:
                self._populate_Cycles(annotation)
                self._populate_CycleIterations(annotation)

    def _get_n_frames(self) -> int:
        """
        Queries and returns the total number of frames in the experiment.
        Used when creating Annotations and Cycles.
        """
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
            UNIQUE("FrameInFile", "FileId")
            )
            """
        db_cursor.execute(sql_create_Frames_table)

        sql_create_Cycles_table = """
            CREATE TABLE IF NOT EXISTS "Cycles" (
            "Id"	INTEGER NOT NULL UNIQUE,
            "AnnotationTypeId"	INTEGER NOT NULL UNIQUE,
            "Structure"	TEXT NOT NULL,
            FOREIGN KEY("AnnotationTypeId") REFERENCES "AnnotationTypes"("Id") ON DELETE CASCADE,
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
            FOREIGN KEY("AnnotationTypeId") REFERENCES "AnnotationTypes"("Id") ON DELETE CASCADE,
            UNIQUE("AnnotationTypeId","Name")
            )
            """
        db_cursor.execute(sql_create_AnnotationTypeLabels_table)

        sql_create_Annotations_table = """
            CREATE TABLE IF NOT EXISTS "Annotations" (
            "FrameId"	INTEGER NOT NULL,
            "AnnotationTypeLabelId"	INTEGER NOT NULL,
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id"),
            FOREIGN KEY("AnnotationTypeLabelId") REFERENCES "AnnotationTypeLabels"("Id") ON DELETE CASCADE,
            UNIQUE("FrameId","AnnotationTypeLabelId")
            )
            """
        db_cursor.execute(sql_create_Annotations_table)

        sql_create_CycleIterations_table = """
            CREATE TABLE IF NOT EXISTS "CycleIterations" (
            "FrameId"	INTEGER NOT NULL,
            "CycleId"	INTEGER NOT NULL,
            "CycleIteration"	INTEGER NOT NULL,
            FOREIGN KEY("CycleId") REFERENCES "Cycles"("Id") ON DELETE CASCADE,
            FOREIGN KEY("FrameId") REFERENCES "Frames"("Id")
            UNIQUE("FrameId","CycleId")
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

    def _populate_Options(self, file_manager: FileManager, volume_manager: VolumeManager):
        """
        Populates the Options table (a dictionary with key - value pairs).
        Learning resources:
            another way of dealing with Errors :
            https://www.w3resource.com/python-exercises/sqlite/python-sqlite-exercise-6.php
        Args:
            file_manager: FileManager object that provides the data to populate the tables.
            volume_manager: VolumeManager object that provides the data to populate the tables.
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

    def _populate_Files(self, file_manager: FileManager):
        """
        Populates the Files table with
        file_name : list with filenames per file (str), and
        num_frames : list with number of frames per file (int)
        Args:
            file_manager: FileManager object that provides the data to populate the tables.
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

    def _populate_Frames(self, frame_manager: FrameManager):
        """
        Populates the Frames table with
        frame_in_file and frame_to_file mapping.

        Learning Resourses:
            Something like
            insert into tab2 (id_customers, value)
            values ((select id from tab1 where customers='john'), 'alfa');
            but in SQLite
            https://www.tutorialspoint.com/sqlite/sqlite_insert_query.htm

        Args:
            frame_manager: FrameManager object that provides the data to populate the tables.
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

    def _populate_Volumes(self, volume_manager: VolumeManager):
        """
        Populates the Volumes table with
        volume_id and slice_in_volume mapping.

        Args:
            volume_manager: VolumeManager object that provides the data to populate the tables.
        """
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

    def _populate_AnnotationTypes(self, annotation: Annotation):
        """
        Populates the AnnotationTypes table with
        annotation.name, annotation.info mapping.

        Args:
            volume_manager: VolumeManager object that provides the data to populate the tables.
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
        n_frames = self._get_n_frames()
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
        assert annotation.cycle is not None, "Annotation is not a Cycle"
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
        n_frames = self._get_n_frames()
        assert n_frames == annotation.n_frames, f"Number of frames in the annotation, {annotation.n_frames}," \
                                                f"doesn't match" \
                                                f" the expected number of frames {n_frames}"
        # get cycle id by annotation type
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT Id FROM Cycles " +
            "WHERE AnnotationTypeId = (SELECT Id FROM AnnotationTypes " +
            "WHERE Name = ?)", (annotation.name,))
        cycle_id = cursor.fetchone()
        assert cycle_id is not None, "Fill out AnnotationTypes and Cycles first."
        cursor.close()

        # prepare rows
        frames = range(n_frames)
        row_data = [(frame + 1, cycle_id[0], iteration)
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

        # get the volumes

    def get_data_dir(self):
        """
        Get data directory
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""SELECT Value FROM Options 
                     WHERE Key = "data_dir" """
            )
            data_dir = cursor.fetchone()[0]
        except Exception as e:
            print(f"Could not get_data_dir because {e}")
            raise e
        finally:
            cursor.close()

        return data_dir

    def get_fpv(self):
        """
        Get frames per volume
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""SELECT Value FROM Options 
                     WHERE Key = "frames_per_volume" """
            )
            fpv = int(cursor.fetchone()[0])
        except Exception as e:
            print(f"Could not get_fpv because {e}")
            raise e
        finally:
            cursor.close()

        return fpv

    def get_fgf(self):
        """
        Get frames per volume
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""SELECT Value FROM Options 
                     WHERE Key = "num_head_frames" """
            )
            fdf = int(cursor.fetchone()[0])
        except Exception as e:
            print(f"Could not get_fgf because {e}")
            raise e
        finally:
            cursor.close()

        return fdf

    def get_options(self):
        """
        Gets the information from the OPTIONS table.
        Returns a dictionary with the Key: Value from the table.
        ( Keys: 'data_dir', 'frames_per_volume', 'num_head_frames', 'num_tail_frames', 'num_full_volumes' )
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""SELECT * FROM Options"""
            )
            options = {}
            for row in cursor.fetchall():
                options[row[0]] = row[1]

        except Exception as e:
            print(f"Could not get_options because {e}")
            raise e
        finally:
            cursor.close()

        return options

    def get_file_names(self):
        """
        Get the file names from the Files table.
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""SELECT FileName FROM Files ORDER BY Id ASC"""
            )
            file_names = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Could not get_file_names because {e}")
            raise e
        finally:
            cursor.close()

        return file_names

    def get_frames_per_file(self):
        """
        Get the file names from the Files table.
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""SELECT NumFrames FROM Files ORDER BY Id ASC"""
            )
            frames_per_file = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Could not get_frames_per_file because {e}")
            raise e
        finally:
            cursor.close()

        return frames_per_file

    def get_volume_list(self):
        """
        Returns a list of all the volumes.
        :return: list of all the volumes
        :rtype: [int]
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT DISTINCT VolumeId FROM Volumes")
            volume_ids = [volume[0] for volume in cursor.fetchall()]
        except Exception as e:
            print(f"Could not get total number of frames from Frames because {e}")
            raise e
        finally:
            cursor.close()

        return volume_ids

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
        # TODO : make one " prepare volumes " for loading?
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
        ids = list_of_int(volume_ids)

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
            labels_ids.append(self._get_Id_from_AnnotationTypeLabels(label_info))
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
            labels_ids.append(self._get_Id_from_AnnotationTypeLabels(label_info))
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

    def get_conditionIds_per_cycle_per_volumes(self, annotation_name):
        """
        Returns a list of condition IDs that correspond to each volume, list of corresponding volumes
        and a count of the volume-condition pairs
        annotation_name: str
        """
        # TODO : check if empty
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT VolumeId, AnnotationTypeLabelId, count(VolumeId) FROM
                    CycleIterations 
                    INNER JOIN Volumes ON CycleIterations.FrameId = Volumes.FrameId 
                    INNER JOIN Annotations ON CycleIterations.FrameId = Annotations.FrameId 
                    WHERE AnnotationTypeLabelId in 
                        (
                        SELECT Id from AnnotationTypeLabels 
                        WHERE AnnotationTypeId = (SELECT Id from AnnotationTypes WHERE Name = ?)
                        )
                    AND CycleId = (SELECT Id from AnnotationTypes WHERE Name = ?)
                    AND CycleIteration = 0
                    GROUP BY VolumeId, AnnotationTypeLabelId
                    ORDER BY VolumeId""", (annotation_name, annotation_name))
            info = cursor.fetchall()
            # TODO : check if empty
            volume_ids = [row[0] for row in info]
            condition_ids = [row[1] for row in info]
            count = [row[2] for row in info]

        except Exception as e:
            print(f"Could not get_conditions_per_cycle_per_volumes because {e}")
            raise e
        finally:
            cursor.close()

        return volume_ids, condition_ids, count

    def get_conditionIds_per_cycle_per_frame(self, annotation_name):
        """
        Returns a list of condition IDs that correspond to each frame, list of corresponding frames.
        annotation_name: str
        """
        # TODO : check if empty
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT CycleIterations.FrameId, AnnotationTypeLabelId FROM
                    CycleIterations 
                    INNER JOIN Annotations ON CycleIterations.FrameId = Annotations.FrameId 
                    WHERE AnnotationTypeLabelId in 
                        (
                        SELECT Id from AnnotationTypeLabels 
                        WHERE AnnotationTypeId = (SELECT Id from AnnotationTypes WHERE Name = ?)
                        )
                    AND CycleId = (SELECT Id from AnnotationTypes WHERE Name = ?)
                    AND CycleIteration = 0
                    ORDER BY CycleIterations.FrameId""", (annotation_name, annotation_name))
            info = cursor.fetchall()
            # TODO : check if empty
            frame_ids = [row[0] for row in info]
            condition_ids = [row[1] for row in info]

        except Exception as e:
            print(f"Could not get_conditionIds_per_cycle_per_frame because {e}")
            raise e
        finally:
            cursor.close()

        return frame_ids, condition_ids

    def get_cycleIterations_per_volumes(self, annotation_name):
        """
        Returns a list of cycleIterations that correspond to each volume, list of corresponding volumes
        and a count of the volume-iteration pairs
        annotation_name: str
        """
        # TODO : check if empty
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT VolumeId, CycleIteration, count(VolumeId) FROM
                    CycleIterations 
                    INNER JOIN Volumes ON CycleIterations.FrameId = Volumes.FrameId 
                    INNER JOIN Annotations ON CycleIterations.FrameId = Annotations.FrameId 
                    WHERE AnnotationTypeLabelId in 
                        (
                        SELECT Id from AnnotationTypeLabels 
                        WHERE AnnotationTypeId = (SELECT Id from AnnotationTypes WHERE Name = ?)
                        )
                    AND CycleId = (SELECT Id from AnnotationTypes WHERE Name = ?)
                    GROUP BY VolumeId
                    ORDER BY VolumeId""", (annotation_name, annotation_name))
            info = cursor.fetchall()
            # TODO : check if empty
            volume_ids = [row[0] for row in info]
            cycle_its = [row[1] for row in info]
            count = [row[2] for row in info]

        except Exception as e:
            print(f"Could not get_cycleIterations_per_volumes because {e}")
            raise e
        finally:
            cursor.close()

        return volume_ids, cycle_its, count

    def get_cycleIterations_per_frame(self, annotation_name):
        """
        Returns a list of cycle iterations that correspond to each frame, list of corresponding frames.
        annotation_name: str
        """
        # TODO : check if empty
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT CycleIterations.FrameId, CycleIteration FROM
                    CycleIterations 
                    INNER JOIN Annotations ON CycleIterations.FrameId = Annotations.FrameId 
                    WHERE AnnotationTypeLabelId in 
                        (
                        SELECT Id from AnnotationTypeLabels 
                        WHERE AnnotationTypeId = (SELECT Id from AnnotationTypes WHERE Name = ?)
                        )
                    AND CycleId = (SELECT Id from AnnotationTypes WHERE Name = ?)
                    ORDER BY CycleIterations.FrameId""", (annotation_name, annotation_name))
            info = cursor.fetchall()
            # TODO : check if empty
            frame_ids = [row[0] for row in info]
            condition_ids = [row[1] for row in info]

        except Exception as e:
            print(f"Could not get_conditionIds_per_cycle_per_frame because {e}")
            raise e
        finally:
            cursor.close()

        return frame_ids, condition_ids

    def get_Names_from_AnnotationTypes(self):
        """
        Returns the names of all the available annotations.
        """
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute("""SELECT Name FROM AnnotationTypes""")
            names = [name[0] for name in cursor.fetchall()]
        except Exception as e:
            print(f"Could not _get_Name_from_AnnotationTypes because {e}")
            raise e
        finally:
            cursor.close()
        return names

    def get_Structure_from_Cycle(self, group):
        """
        Returns cycle Structure if the annotation has a cycle entry or None otherwise.
        """
        group = (group,)
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute("""SELECT Structure FROM Cycles WHERE AnnotationTypeId = 
                             (SELECT Id FROM AnnotationTypes WHERE Name = ?)""", group)
            info = cursor.fetchone()
            if info:
                cycle = info[0]
            else:
                cycle = None
        except Exception as e:
            print(f"Could not get_Structure_from_Cycle because {e}")
            raise e
        finally:
            cursor.close()
        return cycle

    def get_Name_and_Description_from_AnnotationTypeLabels(self, group):
        """
        Returns the Name and description for the labels that correspond to the annotation
        with the provided group name.
        """
        group = (group,)
        names = []
        descriptions = {}

        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute("""SELECT Name, Description FROM AnnotationTypeLabels WHERE AnnotationTypeId = 
                             (SELECT Id FROM AnnotationTypes WHERE Name = ?)""", group)
            info = cursor.fetchall()
            for row in info:
                name = row[0]
                names.append(name)
                descriptions[name] = row[1]
        except Exception as e:
            print(f"Could not get_Name_and_Description_from_AnnotationTypeLabels because {e}")
            raise e
        finally:
            cursor.close()
        return names, descriptions

    def _get_Names_from_AnnotationTypeLabels(self):
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute("""SELECT Name FROM AnnotationTypeLabels ORDER BY Id ASC""")
            names = [name[0] for name in cursor.fetchall()]
        except Exception as e:
            print(f"Could not _get_Name_from_AnnotationTypeLabels because {e}")
            raise e
        finally:
            cursor.close()
        return names

    def get_Id_map_to_Names_from_AnnotationTypeLabels(self):
        """
        Returns a dictionary with Ids as keys and names as values.
        """
        cursor = self.connection.cursor()
        mapping = {}
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute("""SELECT Id, Name FROM AnnotationTypeLabels ORDER BY Id ASC""")
            info = cursor.fetchall()
            for row in info:
                mapping[row[0]] = row[1]
        except Exception as e:
            print(f"Could not get_Id_map_to_Names_from_AnnotationTypeLabels because {e}")
            raise e
        finally:
            cursor.close()
        return mapping

    def _get_Id_from_AnnotationTypeLabels(self, label_info):
        """
        Returns the AnnotationTypeLabels.Id for a label , searching by its name and group name.
        :param label_info: (group, name), where group is the AnnotationType.Name
                            and name is AnnotationTypeLabels.Name
        :type label_info: tuple
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

    def _get_Ids_from_AnnotationTypeLabels(self, group):
        """
        Returns the AnnotationTypeLabels.Id for all labels , searching by their group name.
        :param group: AnnotationType.Name
        :type group: str
        :return: AnnotationLabels.Id
        :rtype: [(int,),(int,) ... ]
        """
        group = (group,)
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f"""SELECT Id FROM AnnotationTypeLabels 
                    WHERE AnnotationTypeId = (SELECT Id from AnnotationTypes WHERE Name = ?)""", group)
            label_ids = cursor.fetchall()
            assert label_ids is not None, f"Could not find labels from group {group} " \
                                          "Are you sure it's been added into the database? "
        except Exception as e:
            print(f"Could not _get_AnnotationLabelIds because {e}")
            raise e
        finally:
            cursor.close()
        return label_ids

    def get_AnnotationTypeLabelId_from_Annotations(self, group):
        """
        Returns the AnnotationTypeLabelIds for all labels in the order according to the FrameId,
         searching by their group name.
        :param group: AnnotationType.Name
        :type group: str
        :return: AnnotationLabels.Id
        :rtype: [int]
        """
        group = (group,)
        cursor = self.connection.cursor()
        try:
            # create a parameterised query with variable number of parameters
            cursor.execute(
                f""" SELECT AnnotationTypeLabelId FROM Annotations WHERE AnnotationTypeLabelId IN
                            (SELECT Id FROM AnnotationTypeLabels 
                            WHERE AnnotationTypeId = (SELECT Id from AnnotationTypes WHERE Name = ?))
                            ORDER BY FrameId ASC""", group)
            label_ids = [label[0] for label in cursor.fetchall()]
            assert label_ids is not None, f"Could not find labels from group {group} " \
                                          "Are you sure it's been added into the database? "
        except Exception as e:
            print(f"Could not _get_AnnotationTypeLabelId_from_Annotations because {e}")
            raise e
        finally:
            cursor.close()
        return label_ids

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

    def delete_annotation(self, name: str):
        """
        Deletes annotation from all the tables.
        Deletion by "ON DELETE CASCADE" from AnnotationTypes, AnnotationTypeLabels, Annotations,
        Cycles, CycleIterations.
        """
        name = (name,)
        # get the volumes
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""DELETE FROM AnnotationTypes 
                            WHERE Name = ?""", name)
        except Exception as e:
            print(f"Could not delete_annotation because {e}")
            raise e
        finally:
            cursor.close()


class DbExporter:
    """
    Transforms the information from the database into the core classes.
    """

    # TODO : make it work for annotations

    def __init__(self, db_reader: DbReader):
        self.db = db_reader

    @classmethod
    def load(cls, file_name: Union[Path, str]):
        """
        Loads a database from a file and initialises a DbExporter.

        Args:
            file_name: full path to a file to database.
        """
        db_reader = DbReader.load(file_name)
        return cls(db_reader)

    def reconstruct_file_manager(self):
        """
        Creates file manager from the database records.
        """
        data_dir = self.db.get_data_dir()
        file_names = self.db.get_file_names()
        frames_per_file = self.db.get_frames_per_file()

        fm = FileManager(data_dir, file_names=file_names, frames_per_file=frames_per_file)
        return fm

    def reconstruct_volume_manager(self):
        """
        Creates volume manager from the database records.
        """
        fm = self.reconstruct_file_manager()
        fpv = self.db.get_fpv()
        fgf = self.db.get_fgf()
        vm = VolumeManager(fpv, FrameManager(fm), fgf=fgf)
        return vm

    def reconstruct_labels(self, group):
        """
        Creates labels corresponding to the specified annotation from the database records.
        """
        state_names, state_info = self.db.get_Name_and_Description_from_AnnotationTypeLabels(group)
        labels = Labels(group, state_names, state_info=state_info)
        return labels

    def reconstruct_timeline(self, group, labels):
        import itertools
        """
        Creates timeline corresponding to the specified annotation from the database records.
        """
        # get the mapping between the names and the Ids
        label_names = self.db.get_Id_map_to_Names_from_AnnotationTypeLabels()
        # get all the labels that were used in the annotation ( does not reconstruct unused Labels )
        labels_per_frame = self.db.get_AnnotationTypeLabelId_from_Annotations(group)

        label_order = []
        duration = []
        for label_id, frames in itertools.groupby(labels_per_frame):
            label_name = label_names[label_id]
            label_order.append(labels.__getattribute__(label_name))
            duration.append(sum(1 for _ in frames))

        timeline = Timeline(label_order, duration)
        return timeline

    def reconstruct_cycle(self, group):
        """
        Creates cycle corresponding to the specified annotation from the database records.
        """
        cycle_json = self.db.get_Structure_from_Cycle(group)
        if cycle_json is not None:
            cycle = Cycle.from_json(cycle_json)
        else:
            cycle = None
        return cycle

    def reconstruct_annotations(self):
        """
        Creates annotations from the database records.
        """
        # get the names of all the available annotations from the db
        annotation_names = self.db.get_Names_from_AnnotationTypes()
        # get the total number of frames in the recording
        n_frames = FrameManager(self.reconstruct_file_manager()).n_frames

        annotations = []
        for group in annotation_names:
            # reconstruct Labels for the group
            labels = self.reconstruct_labels(group)
            # try to get a cycle
            cycle = self.reconstruct_cycle(group)
            # define the annotation type ( Cycle / Timeline)
            if cycle is not None:
                annotation = Annotation.from_cycle(n_frames, labels, cycle)
            else:
                timeline = self.reconstruct_timeline(group, labels)
                annotation = Annotation.from_timeline(n_frames, labels, timeline)

            annotations.append(annotation)

        return annotations
