import inspect


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'TYPE' in cfg # 确保cfg中有TYPE字段
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('TYPE')

    # 根据TYPE的类型，使用不同的方法构建obj_cls
    # 如果obj_type是字符，用registry.get方法，生成配置文件中对应的类
    # 如果obj_type直接就是class，就直接拿来用
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)  # 在这里！！！使用get方法在registry对象中，注册了名为TYPE的类
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    
    # **args表示将字典中的键值对解包，假设 args 是一个字典 {'a': 1, 'b': 2}，那么 **args 就会将它解包成两个关键字参数 a=1 和 b=2
    # obj_cls 是根据配置字典中的 'TYPE' 字段从注册表中获取到的类。
    # 假设配置字典 cfg 的内容为 {'TYPE': 'resnet', 'num_layers': 50, 'pretrained': True}，
    # 那么 obj_cls(**args) 就相当于实例化 resnet 类并传递参数 'num_layers': 50 和 'pretrained': True 给它。这样就可以创建一个名为 resnet 的对象，并将其返回。
    return obj_cls(**args)


def retrieve_from_cfg(cfg, registry):
    """Retrieve a module class from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.

    Returns:
        class: The class.
    """
    assert isinstance(cfg, dict) and 'TYPE' in cfg
    args = cfg.copy()
    obj_type = args.pop('TYPE')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))

    return obj_cls