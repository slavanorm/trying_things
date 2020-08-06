import warnings
import pandas as pd
import json
import time
from pandas import DataFrame as d
import enum
from box import Box
from typing import List, Optional, Union, Dict, NewType
import requests
from enum import Enum

warnings.simplefilter(action="ignore", category=FutureWarning)
user_id = NewType("user_id", str)
resource_id = NewType("resource_id", str)
group_id = NewType("group_id", str)
layer_id = NewType("layer_id", str)
module_id = NewType("module_id", str)
user_or_group_id = NewType("user_or_group_id", str)
scenario_id = NewType("scenario_id", str)
dim_id = NewType("dim_id", str)
fact_id = NewType("fact_id", str)
cube_id = NewType("cube_id", str)


class CredentialsError(Exception):
    pass


class ResTypes(Enum):

    all = 0
    cubes = 500
    profile = 210
    edit = 215
    scenario = 217


class AccessRights:
    df = d()

    @staticmethod
    def get_available_to_user(
        user_or_group_id: Optional[user_or_group_id] = None,
        resource_type: Optional[Union[ResTypes, int]] = None,
    ):
        if resource_type is None:
            resource_type = ResTypes.all
        if type(resource_type) == int:
            resource_type = Box({"value": resource_type})
        if Api._version >= 56:
            if user_or_group_id is None:
                resource_list = d()
                for v in Api.users.dict.values():
                    Api.api(
                        "get_resources_available_to_usr",
                        owner_id=v,
                        resource_type=resource_type.value,
                    )
                    if type(Api.response) == d:
                        resource_list = pd.concat(
                            [resource_list, Api.response]
                        )
                resource_list = resource_list.drop_duplicates()
                if resource_list.empty:
                    resource_list = d()
                Api.resources.df = resource_list
                enddict = Box()
                for ele in set(resource_list.type):
                    enddict[ele] = (
                        resource_list[resource_list.type == ele]
                        .set_index("name")["id"]
                        .to_dict()
                    )
                Api.resources.dict = enddict
            else:
                Api.api(
                    "get_resources_available_to_usr",
                    owner_id=user_or_group_id,
                    resource_type=resource_type.value,
                )
        else:
            print("_version <5.6, only cubes returned")
            Api.api("list_cube")

    @staticmethod
    def _get_208(
        users: Optional[List[user_id]] = None,
        groups: Optional[List[group_id]] = None,
        resources: Optional[List[resource_id]] = None,
    ):
        Api.users.get()
        # todo xx
        Api.api(
            "get_user_access_208",
            user_id=Api.users.dict["admin"],
        )
        df = d.from_dict(Api.response, orient="index")
        df.columns = ["admin"]
        if type(users) == list:
            for ele in users:
                Api.api(
                    "get_user_access_208",
                    user_id=Api.users.dict[ele],
                )
                m2 = d(
                    Api.first_response["permissions"]
                ).set_index("cube_name")[["accessible"]]
                m2.columns = ["user " + ele]
                df = pd.concat([df, m2], axis=1)
        if type(groups) == list:
            for ele in groups:
                Api.api(
                    "get_user_access_208",
                    group_id=Api.users.groups.dict[ele],
                )
                m2 = d(
                    Api.first_response["permissions"]
                ).set_index("cube_name")[["accessible"]]
                m2.columns = ["group " + ele]
                df = pd.concat([df, m2], axis=1)
        df.drop(columns="admin", inplace=True)
        df = df.T[[df.any(axis=1).index]]
        if (
            users is None
            and groups is None
            and resources is None
        ):
            Api.users.access.df = df
        else:
            Api.response = df

    @staticmethod
    def _give_or_take(
        resources: List[resource_id],
        give: bool = False,
        permissions: str = 0,
        users: Optional[List[user_id]] = None,
        groups: Optional[List[group_id]] = None,
    ):
        if Api._version == 54:
            if type(users) == list:
                for ele1 in users:
                    if give:
                        Api.api(
                            [208, 22],
                            user_id=a.users[ele1],
                            permissions_set=[
                                {
                                    "cube_id": a.cubes[ele2],
                                    "accessible": True,
                                    "dimensions_denied": [],
                                    "facts_denied": [],
                                }
                                for ele2 in resources
                            ],
                        )
                    else:
                        Api.api(
                            [208, 22],
                            user_id=a.users[ele1],
                            permissions_set=[
                                {
                                    "cube_id": a.cubes[ele2],
                                    "accessible": False,
                                    "dimensions_denied": [],
                                    "facts_denied": [],
                                }
                                for ele2 in resources
                            ],
                        )
            if type(groups) == list:
                for ele1 in groups:
                    if give:
                        Api.api(
                            [208, 22],
                            group_id=a.groups[ele1],
                            permissions_set=[
                                {
                                    "cube_id": a.cubes[ele2],
                                    "accessible": True,
                                    "dimensions_denied": [],
                                    "facts_denied": [],
                                }
                                for ele2 in resources
                            ],
                        )
                    else:
                        Api.api(
                            [208, 22],
                            group_id=a.groups[ele1],
                            permissions_set=[
                                {
                                    "cube_id": a.cubes[ele2],
                                    "accessible": False,
                                    "dimensions_denied": [],
                                    "facts_denied": [],
                                }
                                for ele2 in resources
                            ],
                        )
        else:
            myinput = []
            if type(groups) == list:
                myinput += [
                    Api.users.groups.dict[v] for v in groups
                ]
            if type(users) == list:
                myinput += [Api.users.dict[v] for v in users]
            elif groups == None:
                myinput = [Api.users.dict[Api._login]]
            if type(permissions) == str:
                p_dict = {
                    "read": 0,
                    "share": 1,
                    "full": 4294967295,
                }
                permissions = p_dict[permissions]
            else:
                permissions = 0
            for ele in myinput:
                for ele1 in resources:
                    if give:
                        Api.api(
                            [224, 6],
                            recipient_id=ele,
                            resources=[
                                {
                                    "id": ele1,
                                    "permissions": permissions,
                                }
                            ],
                        )
                    else:
                        Api.api(
                            [224, 0],
                            owner_id=ele,
                            resource_id=ele1,
                        )

    @classmethod
    def give(
        cls,
        resources: List[resource_id],
        users: Optional[List[user_id]] = None,
        groups: Optional[List[group_id]] = None,
        permissions: Optional[str] = None,
    ):
        cls._give_or_take(
            give=True,
            resources=resources,
            users=users,
            groups=groups,
            permissions=permissions,
        )

    @classmethod
    def take(
        cls,
        resources: List[resource_id],
        users: Optional[List[user_id]] = None,
        groups: Optional[List[group_id]] = None,
    ):
        cls._give_or_take(
            give=False,
            resources=resources,
            users=users,
            groups=groups,
        )


class Groups:
    df = d()
    dict = Box()

    @staticmethod
    def get():
        Api.api("get_groups")


class Users:
    access = AccessRights()
    groups = Groups()
    df = d()
    dict = Box()

    @staticmethod
    def get():
        Api.api("get_users")


class Layers:
    df = d()
    dict = Box()

    @staticmethod
    def get():
        Api.api("get_session_lr")

    @staticmethod
    def create():
        Api.api("create_lr")
        pass

    @staticmethod
    def rename(name: str, layer_id: Optional[layer_id] = None):
        if layer_id == None:
            Api.api("rn_lr", name=name)
        else:
            Api.api("rn_lr", layer_id=layer_id, name=name)

    @staticmethod
    def close(layer_id: Optional[layer_id] = None):
        if layer_id is None:
            Api.api("close_lr")
        else:
            Api.api("close_lr", layer_id=layer_id)

    @staticmethod
    def set(layer_id: layer_id):
        Api._layer_id = layer_id


class Modules:
    df = d()

    @staticmethod
    def get():
        Api.api("get_module")

    @staticmethod
    def set(module_id: module_id):
        Api._module_id = module_id


class Resources:
    df = d()
    dict = Box()

    # todo сложить все в users.access.get

    @staticmethod
    def get_resource_readers(resource_id: resource_id):
        if Api._version >= 56:
            Api.api(
                "get_resource_readers", resource_id=resource_id
            )
        else:
            print("_version <5.6, not working")

    @staticmethod
    def open(
        cube_id: Optional[cube_id] = None,
        layer_id: Optional[layer_id] = None,
        module_id: Optional[module_id] = None,
    ):
        if Api._module_id == "":
            Api.layers.get()
            Api.api(
                [210, 4],
                layer_id=list(a.layers.dict.values())[0],
            )
        args = {
            "cube_id": cube_id,
            "layer_id": layer_id,
            "module_id": module_id,
        }
        pop = []
        for k, v in args.items():
            if v == None:
                pop.append(k)
        if len(pop) > 0:
            for ele in pop:
                args.pop(ele)
        Api.api("open_cube", **args)

    @staticmethod
    def run(scenario_id: scenario_id = None):
        Api.api("runsc", script_id=scenario_id)


class Data:
    df = d()

    @staticmethod
    def get():
        Api.api(
            "getdata",
            params=[
                "no_cols",
                "rows_dot",
                "cols_dot",
                "no_total",
            ],
        )


class Dims:
    df = d()
    dict = Box()

    @staticmethod
    def group_left():
        """
        groups all visible dims into 1 row

        """
        # todo adj
        api([506, 19], position=1, line=0, level=0)

    @staticmethod
    def group_top():
        """
        groups all visible dims into 1 row

        """
        # todo adj
        api([506, 19], position=2, line=0, level=0)

    @staticmethod
    def select_left():
        # todo adj
        Api.api([506, 17], position=1, line=0, level=0)

    @staticmethod
    def select_top():
        # todo adj
        Api.api([506, 17], position=2, line=0, level=0)

    @staticmethod
    def top(dim_id: Union[List[dim_id], dim_id]):
        if type(dim_id) == list:
            for ele in reversed(dim_id):
                Api.api("move_dim", position=2, level=0, id=ele)
        else:
            Api.api("move_dim", position=2, level=0, id=dim_id)

    @staticmethod
    def left(dim_id: Union[List[dim_id], dim_id]):
        if type(dim_id) == list:
            for ele in reversed(dim_id):
                Api.api("move_dim", position=1, level=0, id=ele)
        else:
            Api.api("move_dim", position=1, level=0, id=dim_id)

    @staticmethod
    def hide(dim_id: Union[List[dim_id], dim_id]):
        if type(dim_id) == list:
            for ele in reversed(dim_id):
                Api.api("move_dim", position=0, level=0, id=ele)
        else:
            Api.api("move_dim", position=0, level=0, id=dim_id)

    @staticmethod
    def get():
        Api.api("get_dims")

    @staticmethod
    def copy(dim_id: dim_id):
        Api.api("copy_dim", id=dim_id)

    @staticmethod
    def name(dim_id: dim_id, name: str):
        Api.api("rename_dim", id=dim_id, name=name)

    @staticmethod
    def filter(
        dim_name_and_filter: Dict[dim_id, Union[str, List[str]]]
    ):
        for k, v in dim_name_and_filter.items():
            api("filter_drop", dimension=Api.dims.dict[k])
            # run filter
            api(
                "filter_run",
                dimension=Api.dims.dict[k],
                pattern="",
                num=99,
            )
            # get filter data
            api(
                [504, 1],
                extrakw=["from", 0],
                dimension=Api.dims.dict[k],
                num=99,
            )
            if type(v) == list:
                marks = [
                    (value in v) for value in a.response["data"]
                ]
            else:
                marks = [(kk == v) for kk in a.response["data"]]
            marks = [1 if z else 0 for z in marks]
            # select
            api(
                [504, 9],
                dimension=Api.dims.dict[k],
                marks=marks,
                extrakw=["from", 0],
            )
            # apply filter
            api([504, 17], dimension=Api.dims.dict[k])


class FactTypes(Enum):
    sum = 0
    percent = 1
    rank = 2
    count_distinct = 3
    avg = 4
    deviation = 5
    min = 6
    max = 7
    diff = 8
    diff_in_percent = 9
    cumulative = 10
    ABC = 11
    median = 12


class Facts:
    df = d()
    dict = Box()

    @staticmethod
    def filter():
        # todo this wrong
        pass

    @staticmethod
    def get():
        Api.api("get_fact_list")

    @staticmethod
    def type(fact_id: fact_id, fact_type: FactTypes):
        Api.api(
            "set_fact_type", fact=fact_id, type=fact_type.value
        )

    @staticmethod
    def copy(fact_id: fact_id):
        Api.api("copy_fact", fact=fact_id)

    @staticmethod
    def name(name: str, fact_id: fact_id):
        api("fact_rename", fact=fact_id, name=name)

    @staticmethod
    def fold(fact_id: fact_id):
        api("tg_fact_visible", fact=fact_id, is_visible=False)

    @staticmethod
    def unfold(fact_id: fact_id):
        api("tg_fact_visible", fact=fact_id, is_visible=True)


class Api:
    """
    this works completely in class methods, variables.
    no instance methods used.
    """

    @classmethod
    def _pickup(cls, ans: dict, mode: str, params=0):
        """pick up data from response"""
        cls._pretty(ans)
        cls.response = ans
        cls.first_response = ans
        if mode == "auth":
            cls._session = ans["session_id"]
            cls._manager = ans["manager_uuid"]
            cls._version = int(
                "".join(cls.response["version"].split(".")[:2])
            )
        elif mode == "get_module":
            try:
                cls._module_id = ans["layer"]["module_descs"][0][
                    "uuid"
                ]
            except:
                cls._module_id = ans["layer"]["uuid"]
        elif mode == "runsc":
            cls._layer_id = ans["layer"]["uuid"]
            while True:
                time.sleep(3)
                api("waitsc")
                print("{:.3}%".format(cls._prog_bar))
                if cls._prog_bar > 98:
                    break
        elif mode == "waitsc":
            cls._prog_bar = 100 * (
                ans["finished_steps_count"]
                / ans["total_steps_count"]
            )
        elif mode == "seedata":
            cls.nrow = ans["total_row"]
            # todo this wrong
            if ans["show_inter_total"] is True:
                cls.nrow -= 1
            cls.ncol = ans["total_col"]
        elif mode == "getdata":
            cls.api("seedata")
            cls.first_response = ans
            df = d(ans["data"])
            labels = []
            for ele in range(cls.nrow):
                part = []
                for ele1 in range(len(ans["left"][ele])):
                    if ans["left"][ele][ele1]["type"] == 2:
                        part.append(
                            (ans["left"][ele][ele1]["value"])
                        )
                    if ans["left"][ele][ele1]["type"] == 5:
                        part.append("subtotal")
                if part != []:
                    labels.append(part)
            if "no_cols" in params:
                df.insert(0, "labels", labels)
                cls.response = df
            else:
                rows_legend = []
                for ele in ans["left_dims"]:
                    for k, v in cls.dims.items():
                        if v == ele:
                            rows_legend.append(k)
                cols_legend = []
                for ele in ans["top_dims"]:
                    for k, v in cls.dims.items():
                        if v == ele:
                            cols_legend.append(k)
                myrowstop = {}
                myrowstop2 = {}
                for j in range(len(ans["top"])):
                    myrowstop[j] = {}
                    for i in range(len(ans["top"][j])):
                        myrowstop[j][i] = []
                        if ans["top"][j][i]["type"] == 2:
                            myrowstop[j][i] = ans["top"][j][i][
                                "value"
                            ]
                        if ans["top"][j][i]["type"] == 1:
                            myrowstop[j][i] = myrowstop[j][i - 1]
                        if ans["top"][j][i]["type"] == 5:
                            myrowstop[j][i] = "Total"
                        if ans["top"][j][i]["type"] == 4:
                            for k, v in cls.facts.dict.items():
                                if (
                                    v
                                    == ans["top"][j][i][
                                        "fact_id"
                                    ]
                                ):
                                    myrowstop[j][i] = k
                    myrowstop2[j] = list(myrowstop[j].values())

                if "rows_dot" in params:
                    rows_legend = []
                    for ele in labels:
                        rows_legend.append(".".join(ele))
                    s1 = pd.concat(
                        [d(rows_legend), df],
                        axis=1,
                        ignore_index=True,
                    )
                    s1.set_index(0, inplace=True)
                else:
                    s1 = pd.concat(
                        [d(labels), df],
                        axis=1,
                        ignore_index=True,
                    )
                    s1.set_index(
                        [i for i in range(d(labels).shape[1])],
                        inplace=True,
                    )

                s1.columns = [i for i in range(s1.shape[1])]
                if "cols_dot" in params:
                    s2 = []
                    for ele in d(myrowstop2).T.columns:
                        s2.append(
                            ".".join(
                                d(myrowstop2).T[ele].to_list()
                            )
                        )
                    s2 = d(s2).T
                    s2.rename(index={0: cols_legend})
                else:
                    s2 = d(myrowstop2).T  # col legend
                    for i in range(len(cols_legend)):
                        s2 = s2.rename(index={i: cols_legend[i]})
                df = pd.concat([s2, s1])
                if "cols_dot" in params:
                    df.columns = df.iloc[0]
                    df = df.drop(0)
                if "no_total" in params:
                    df = df.T[
                        ~df.columns.str.contains("Total")
                    ].T
                    df = df[~(df.index == "subtotal")]
                cls.response = df
            cls.data[cls.name] = cls.response
        elif mode == "list_cube":
            cls.resources.df = pd.DataFrame(ans["cubes"])
            cls.resources.dict = Box()
            cls.resources.dict["cubes"] = (
                cls.resources.df[["name", "uuid"]]
                .set_index("name")
                .to_dict()["uuid"]
            )
        elif mode == "get_fact_list":
            cls.response = pd.DataFrame(ans)["facts"].to_list()
            cls.facts.df = pd.io.json.json_normalize(
                cls.response
            )
            cls.facts.dict = Box(
                cls.facts.df.set_index("name")["id"].to_dict()
            )
        elif mode == "get_session_lr":
            if cls.response["layers"] == []:
                api("create_lr")
            cls.layers.df = pd.DataFrame(ans["layers"])

            cls.layers.dict = Box(
                cls.layers.df[["name", "uuid"]]
                .set_index("name")
                .to_dict()["uuid"]
            )

            cls._layer_id = ans["layers"][0]["uuid"]

            b = cls.layers.df
            b1 = b[["uuid", "name", "module_descs"]].to_dict()
            modules = []
            for k, v in b1["module_descs"].items():
                for ele in v:
                    modules.append(
                        dict(
                            layer_name=(b1["name"][k]),
                            layer_uuid=(b1["uuid"][k]),
                            module_uuid=ele.pop("uuid"),
                            **ele,
                        )
                    )
            cls.modules.df = d(modules)
        elif mode == "get_dims":
            cls.response = pd.DataFrame(ans)[
                "dimensions"
            ].to_list()
            cls.dims.df = pd.io.json.json_normalize(cls.response)
            cls.dims.dict = Box(
                cls.dims.df.set_index("name")["id"].to_dict()
            )
        elif mode == "create_lr":
            cls._layer_id = ans["layer"]["uuid"]
            pass
        elif mode == "open_cube":
            cls._module_id = ans["module_desc"]["uuid"]
            pass
        elif mode == "get_groups":
            df = d(cls.response["groups"]).set_index("name")
            cls.users.groups.dict = Box(df["uuid"].to_dict())
            cls.users.groups.df = df
        elif mode == "get_users":
            df = d(cls.response["users"]).set_index("login")
            cls.users.dict = Box(df["uuid"].to_dict())
            cls.users.df = df
        elif mode == "get_resources_available_to_usr":
            df = d(cls.response["resources"])
            try:
                df.replace(
                    {
                        "type": {
                            500: "cube",
                            210: "profile",
                            215: "edit",
                            217: "scenario",
                        }
                    },
                    inplace=True,
                )
                df = df[df["type"] != "edit"]
            except:
                pass
            cls.response = df
        elif mode == "get_user_access_208":
            df = d(cls.response["permissions"]).set_index(
                "cube_name"
            )
            cls.response = df["cube_id"].to_dict()
        elif mode == "get_resource_readers":
            cls.response = (
                d(cls.response["owners"])
                .set_index("login")["id"]
                .to_dict()
            )

    def __setattr__(self, key, value):
        # this overrides instance to class editing
        setattr(self.__class__, key, value)

    @classmethod
    def api(cls, mode: Union[str, List[int]], **kwargs):
        params_allowed = {
            "getdata": [
                "no_cols",
                "rows_dot",
                "cols_dot",
                "no_total",
            ],
            "timeout": "",
            "params": "",
        }
        if "timeout" in kwargs:
            t = kwargs.pop("timeout")
        else:
            t = 1
        if "params" in kwargs:
            p = kwargs.pop("params")
        else:
            p = []
        query = cls._make_query(mode, **kwargs)
        if cls._verbose > 1:
            print("query:", mode, query)
        ans = requests.post(cls._site, json=query, timeout=t)
        ans = json.loads(ans.text)["queries"][0]["command"]
        cls._error_handle(ans)
        try:
            cls._pickup(ans, mode, p)
        except Exception as e:
            print(e)

    @classmethod
    def _make_query(cls, mode: str, **kwargs):
        """add required info to query"""
        query = {
            "state": 0,
            "session": "",
            "queries": [{"uuid": "", "command": {}}],
        }
        if mode == "getdata":
            cls.api("seedata")
        if mode == "seedata":
            query["queries"][0]["command"].update(
                {
                    "from_row": 0,
                    "from_col": 0,
                    "num_row": 1,
                    "num_col": 1,
                }
            )
        if mode == "getdata":
            query["queries"][0]["command"].update(
                {
                    "from_row": 0,
                    "from_col": 0,
                    "num_row": cls.nrow,
                    "num_col": cls.ncol,
                }
            )
        if mode == "runsc":
            query["queries"][0]["command"].update(
                {"_script_id": cls._script_id}
            )
        if cls._session != "":
            query["session"] = cls._session
        if cls._layer_id != "":
            query["queries"][0]["command"].update(
                {"layer_id": cls._layer_id}
            )
        # plm_type_code, state
        if type(mode) == list:
            code = mode[0]
            state = mode[1]
        elif cls._version == 56:
            code, state = cls._api_dict_56[mode]
        else:
            code, state = cls._api_dict[mode]
        query["queries"][0]["command"].update(
            {"plm_type_code": code, "state": state}
        )
        query = cls._uuid(query, code, state)
        # add kw that you cant normally
        if "extrakw" in kwargs:
            z = {kwargs["extrakw"][0]: kwargs["extrakw"][1]}
            query["queries"][0]["command"].update(z)
            kwargs.pop("extrakw")
        # drop kw that you cant normally
        if "dropkw" in kwargs:
            for ele in kwargs["dropkw"]:
                query["queries"][0]["command"].pop(ele)
            kwargs.pop("dropkw")
        query["queries"][0]["command"].update(**kwargs)
        return query

    @classmethod
    def _pretty(cls, json_dict):
        """format JSON print"""
        if cls._verbose > 0:
            print(
                (
                    json.dumps(
                        json_dict, indent=2, ensure_ascii=False
                    )
                ),
                "\n",
            )

    @classmethod
    def _error_handle(cls, ans: dict):
        if not cls.error_mute:
            if "error" in ans:
                if "message" in ans["error"]:
                    print(ans["error"]["message"])
                if ans["error"]["code"] == 270:
                    raise CredentialsError(
                        f"Bad credentials for {cls._login}",
                    )
                if ans["error"]["code"] == 204:
                    raise Exception(
                        f"User {cls._login} needs admin access",
                    )

    @classmethod
    def _uuid(cls, query: dict, code: int, state: int):
        """cases to set uuid"""
        if str(code)[0] == "2" or cls._module_id == "":
            query["queries"][0].update({"uuid": cls._manager})
        elif str(code)[0] == 5:
            query["queries"][0].update({"uuid": cls._cube_id})
        else:
            try:
                query["queries"][0].update(
                    {"uuid": cls._module_id}
                )
            except Exception as e:
                print(e)
        return query

    @classmethod
    def auth(cls, login: str, password: str, site: str = None):
        cls._login = login
        cls._password = password

        if site is None:
            if cls._site is None:
                raise Exception("please define site")
        else:
            cls._site = site

        cls.api(
            "auth",
            login=login,
            passwd=password,
            locale=0,
            timeout=5,
        )
        cls.error_mute = True
        cls.users.get()
        cls.users.groups.get()
        cls.layers.get()
        cls.modules.get()
        cls.users.access.get_available_to_user()
        cls.error_mute = False

    # settings
    _login = ""
    _password = ""
    _version = 0
    _verbose = 0  # 1 for receive, 2 for send
    _session = ""
    _manager = ""
    _layer_id = ""
    _script_id = ""
    _module_id = ""
    _cube_id = ""
    _ans = ""
    _prog_bar = 0  # scenario progress
    response = ""
    first_response = ""
    error_mute = False

    _api_dict_56 = {
        "auth": [205, 2],
        "create_lr": [210, 1],
        "get_session_lr": [210, 21],
        "list_cube": [208, 1],
        "open_cube": [208, 16],
        "get_module": [210, 2],
        "rn_lr": [210, 14],
        "get_usr_list": [206, 3],
        "runsc": [217, 4],
        "waitsc": [217, 14],
        "list_sc": [217, 6],
        "get_view": [506, 1],
        "seedata": [506, 1],
        "getdata": [506, 1],
        "copy_dim": [502, 10],
        "new_value": [506, 26],
        "get_dims": [502, 1],
        "copy_fact": [503, 3],
        "new_row": [506, 22],
        "get_fact_list": [503, 1],
        "move_dim": [502, 3],
        "fact_rename": [503, 24],
        "set_fact_type": [503, 9],
        "tg_fact_visible": [503, 12],
        "rename_dim": [502, 5],
        "close_lr": [210, 10],
        "unfold_dim": [506, 13],  # error in api
        "filter_drop": [504, 19],
        "filter_run": [504, 2],
        "tg_subtotal": [506, 21],
        "get_groups": [218, 3],
        "get_users": [206, 4],
        "get_resources_available_to_usr": [
            224,
            10,
        ],  # 5.6 tested
        "get_user_access_208": [208, 18],
        "get_resource_readers": [224, 8],
    }
    _api_dict = {
        "auth": [205, 2],
        "create_lr": [210, 1],
        "get_session_lr": [210, 21],
        "list_cube": [208, 1],
        "open_cube": [208, 16],
        "get_module": [210, 2],
        "rn_lr": [210, 14],
        "get_usr_list": [206, 3],
        "runsc": [217, 4],
        "waitsc": [217, 14],
        "list_sc": [217, 6],
        "get_view": [506, 1],
        "seedata": [506, 1],
        "getdata": [506, 1],
        "copy_dim": [502, 9],
        "new_value": [506, 26],
        "get_dims": [502, 1],
        "copy_fact": [503, 3],
        "new_row": [506, 22],
        "get_fact_list": [503, 1],
        "move_dim": [502, 3],
        "fact_rename": [503, 24],
        "set_fact_type": [503, 9],
        "tg_fact_visible": [503, 12],
        "rename_dim": [502, 5],
        "close_lr": [210, 10],
        "unfold_dim": [506, 13],  # error in api
        "filter_drop": [504, 19],
        "filter_run": [504, 2],
        "tg_subtotal": [506, 21],
        "get_groups": [218, 3],
        "get_users": [206, 3],
        "get_resources_available_to_usr": [
            224,
            10,
        ],  # 5.6 tested
        "get_user_access_208": [208, 18],
    }

    # inherit
    users = Users()
    layers = Layers()
    modules = Modules()
    resources = Resources()
    data = Data()
    dims = Dims()
    facts = Facts()
