import os
import sys
import time
import threading
import pprint
import math
import argparse
import json
import xml.etree.ElementTree
# from typing import Self
from pydantic import BaseModel

def log(buf, *args):
    print(buf % args, file=sys.stderr)

def fmt_cs(base, l):
    return "{{0:0{0}X}}".format(l).format(-base % (16 ** l))

def len_cs(nlen):
    mid = nlen % 16 + (nlen // 16) % 16 + (nlen // 16**2) % 16
    return "{0:X}".format((-mid + 16) % 16)

def build_cmd(code, data, addr="00"):
    assert len(code) == 2
    l = len(data)
    cmd = f"20{addr}46{code}{len_cs(l)}{l:03X}{data}"
    csum = sum(ord(c) for c in cmd)
    return f"{cmd}{fmt_cs(csum, 4)}"

class StreamReader:
    def __init__(self, buffer, offset=0):
        self._buffer = buffer
        self._offset = offset

    def read(self, n):
        res = self._buffer[self._offset:self._offset+n]
        self.skip(n)
        return res

    def skip(self, n):
        self._offset += n

    def read_hex(self, n):
        shex = self.read(n)
        return int(shex, 16)

    def remainder(self):
        return self._buffer[self._offset:]

    def remaining(self):
        return len(self._buffer) - self._offset


class SeplosProtocol(BaseModel):

    class IntParam(BaseModel):
        name: str
        unit_name: str
        index: int
        byte_num: int
        scale: float

        def encode(self, value):
            if self.unit_name == '\u2103':
                # kelvin -> C
                value += 273.1
            scaled = int(value / self.scale)
            if self.byte_num == 2 and scaled < 0:
                scaled += 2 ** (8 * self.byte_num)
            return (f"{{scaled:0{self.byte_num*2}X}}"
                    .format(scaled=scaled))

        def decode(self, encoded):
            decoded = int(encoded, 16)
            if self.unit_name == '\u2103':
                decoded -= 2731
            if self.byte_num == 2 and decoded >= 2 ** (8 * self.byte_num - 1):
                decoded -= 2 ** (8 * self.byte_num)
            return decoded * self.scale

        def from_str(self, svalue):
            # print(f'parsing {svalue} {self.scale}')
            if self.scale == 1:
                return int(svalue)
            else:
                return float(svalue)

        def fmt(self, value):
            digits = round(math.log(self.scale, 0.1))
            print(f"{digits} {value}")
            return f"{{value:.{digits}f}}".format(value=value)

    class BitParam(BaseModel):
        name: str
        byte_index: int
        bit_index: int

    class Signal(BitParam):
        typ: str

    int_params: list[IntParam]
    bit_param_bytes: int
    bit_params: list[BitParam]

    signal_bit_params: list[Signal]
    signal_modes: dict[int, str]

    @classmethod
    def from_proto_file(cls, proto_file: str):
        ET = xml.etree.ElementTree
        tree = ET.parse(proto_file)
        root = tree.getroot()

        int_params = []
        for int_para in root.find('int_para_Group'):
            name = int_para.findtext('Name')
            unit_name = int_para.findtext('UnitName')
            index = int_para.findtext('ParaIndex')
            byte_num = int_para.findtext('ByteNum')
            scale = int_para.findtext('Scale')

            int_params.append(cls.IntParam(
                name=name,
                unit_name=unit_name,
                index=int(index, 16),
                byte_num=byte_num,
                scale=scale,
            ))

        bit_params = []
        bit_group = root.find('bit_para_Group')
        for bit_param in bit_group.findall('bit_para'):
            name = bit_param.findtext('Name')
            byte_index = bit_param.findtext('ByteIndex')
            bit_index = bit_param.findtext('BitIndex')

            bit_params.append(cls.BitParam(
                name=name,
                byte_index=byte_index,
                bit_index=bit_index,
            ))

        # teleSignal -- last 2 blocks
        signal_bit_params = []
        signal_group = root.find('teleSignal_Group')

        signal_bit_group = signal_group.findall('teleSignal_block')[-2]
        for signal in signal_bit_group.findall('teleSignal'):
            signal_bit_params.append(cls.Signal(
                name=signal.findtext('Name'),
                typ=signal.findtext('Type'),
                bit_index=signal.findtext('BitIndex'),
                byte_index=signal.findtext('ByteIndex'),
            ))

        signal_modes = {}
        signal_mode_group = signal_group.findall('teleSignal_block')[-1]
        for mode_map in signal_mode_group.findall('modeText'):
            value = mode_map.findtext('value')
            text = mode_map.findtext('text')
            signal_modes[int(value, 16)] = text

        return cls(int_params=int_params,
                   bit_param_bytes=bit_group.findtext('bitParaByteNum'),
                   bit_params=bit_params,
                   signal_bit_params=signal_bit_params,
                   signal_modes=signal_modes)

    def get_param(self, name=None, idx=None) -> IntParam:
        maymatch = lambda a, b: a == b if a is not None else True
        pf = [p for p in self.int_params
              if maymatch(name, p.name) and maymatch(idx, p.index)]
        if len(pf) != 1:
            raise ValueError(f"Can not find a single int_param matching name={name} idx={idx}")
        return pf[0]

def str_decode(inp):
    # return re.sub(r'..', lambda x: chr(int(x, 16)), inp)
    print(f"__{inp}__")
    return ''.join([chr(int(inp[i*2:i*2+2], 16))
                    for i in range(len(inp) // 2)])

def str_encode(inp):
    return ''.join([f"{ord(c):02X}" for c in inp])

class SeplosParams(BaseModel):
    protocol: SeplosProtocol
    numeric_params: list[float]
    bit_groups: list[int]
    model_name: str

    @classmethod
    def from_save_file(cls, protocol: SeplosProtocol, save_file: str) -> 'SeplosParams':
        ET = xml.etree.ElementTree
        tree = ET.parse(save_file)
        root = tree.getroot()

        numeric_params = []
        def reserve(n):
            if n >= len(numeric_params):
                numeric_params.extend([None] * (n+1 - len(numeric_params)))

        for param in root.findall('parameter'):
            name = param.findtext('Name')
            value = param.findtext('Value')
            unit = param.findtext('Unit')

            pp = protocol.get_param(name=name)
            reserve(pp.index)
            numeric_params[pp.index] = pp.from_str(value)

        bit_groups = []
        for param in root.findall('param_bit_group'):
            name = param.findtext('Name')
            value = param.findtext('Value')
            bit_groups.append(int(value, 16))

        model_name = root.findtext('modualName')

        return cls(
            protocol=protocol,
            numeric_params=numeric_params,
            bit_groups=bit_groups,
            model_name=model_name,
        )

    @classmethod
    def from_msg_body(cls, protocol, body):
        sr = StreamReader(body, 2)

        base = 0
        numeric_params = [
            protocol.get_param(idx=base+i).decode(sr.read(4))
            for i in range(sr.read_hex(2))
        ]
        base = len(numeric_params)
        numeric_params.extend([
            protocol.get_param(idx=base+i).decode(sr.read(2))
            for i in range(sr.read_hex(2))
        ])
        bit_groups = [
            sr.read_hex(2) for i in range(sr.read_hex(2))
        ]
        # model_name_enc = sr.remainder()
        # if len(model_name_enc) != 16:
        #     raise Exception("expected 16 characters model name")

        # model_name = ''.join([chr(int(model_name_enc[i*2:i*2+2], 16))
        #                       for i in range(8)])

        return cls(
            protocol=protocol,
            numeric_params=numeric_params,
            bit_groups=bit_groups,
            model_name=str_decode(sr.remainder()),
        )

    def to_msg_body(self):
        chunks_2b = []
        chunks_1b = []
        bit_groups = []

        for i, param in enumerate(self.numeric_params):
            if param is None:
                raise ValueError(f"non-specified argument at index {i}")
            pspec = self.protocol.get_param(idx=i)
            chunk = pspec.encode(param)
            # chunks.append(chunk)
            # log(f"{i} {chunk} {param}")
            if pspec.byte_num == 2:
                chunks_2b.append(chunk)
            else:
                chunks_1b.append(chunk)

        # import pdb; pdb.set_trace()
        for param in self.bit_groups:
            bit_groups.append(f"{param:02X}")

        len_encode = lambda x: f"{len(x):02X}{''.join(x)}"
        pcs = [
            '00',
            len_encode(chunks_2b),
            len_encode(chunks_1b),
            len_encode(bit_groups),
            str_encode(self.model_name),
        ]
        # print("\n".join(pcs))
        return ''.join(pcs)

    def to_save_file(self, save_file):
        ET = xml.etree.ElementTree
        paraGroup = ET.Element('paraGroup')
        filetype = ET.SubElement(paraGroup, 'filetype').text = 'Parameter'
        modualName = ET.SubElement(paraGroup, 'modualName').text = self.model_name
        for int_prop in self.protocol.int_params:
            parameter = ET.SubElement(paraGroup, 'parameter')
            value = self.numeric_params[int_prop.index]
            ET.SubElement(parameter, 'Name').text = int_prop.name
            ET.SubElement(parameter, 'Value').text = int_prop.fmt(value)
            ET.SubElement(parameter, 'Unit').text = int_prop.unit_name

        for i, value in enumerate(self.bit_groups):
            bit_group = ET.SubElement(paraGroup, 'param_bit_group')
            ET.SubElement(bit_group, 'Name').text = f"BitGroup{i}"
            ET.SubElement(bit_group, 'Value').text = f"{value:02X}"

        ET.indent(paraGroup)
        with open(save_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(ET.tostring(paraGroup, encoding='unicode'))

class MeterReply(BaseModel):
    ncells: int
    cell_mv: list[int]
    ntemps: int
    temp_c: list[float]

    charge_discharge_a: float
    total_voltage_v: float
    residual_capacity_ah: float
    batt_capacity_ah: float
    soc_percent: float
    rated_capacity_ah: float
    ncycles: int
    soh_percent: float
    port_voltage_v: float

    @classmethod
    def from_reply(cls, data, protocol=None):
        # data = buf[12:-4]
        s = StreamReader(data, 4)
        ncells = s.read_hex(2)
        cell_mv = [s.read_hex(4) for _ in range(ncells)]
        ntemps = s.read_hex(2)
        temp_map = lambda rt: (rt - 2731)/10
        temp_c = [temp_map(s.read_hex(4)) for _ in range(ntemps)]
        sign_map = lambda v: v if v < 2**15 else v - 2**16

        return cls(
            ncells=ncells,
            cell_mv=cell_mv,
            ntemps=ntemps,
            temp_c=temp_c,
            charge_discharge_a=sign_map(s.read_hex(4)) / 100,
            total_voltage_v=s.read_hex(4) / 100,
            residual_capacity_ah=s.read_hex(4) / 100,
            _extra=s.read_hex(2), # this is the numer of remaining fields
            batt_capacity_ah=s.read_hex(4) / 100,
            soc_percent=s.read_hex(4) / 10,
            rated_capacity_ah=s.read_hex(4) / 100,
            ncycles=s.read_hex(4),
            soh_percent=s.read_hex(4) / 10,
            port_voltage_v=s.read_hex(4) / 100,
        )


class SignalReply(BaseModel):
    cells: list[int]
    temps: list[int]
    current: int
    total_voltage: int
    flag_bytes: list[int]
    mode: int

    @classmethod
    def from_reply(cls, data):
        s = StreamReader(data, 4)
        ncells = s.read_hex(2)
        cells = [s.read_hex(2) for _ in range(ncells)]
        ntemps = s.read_hex(2)
        temps = [s.read_hex(2) for _ in range(ntemps)]
        current = s.read_hex(2)
        total_voltage = s.read_hex(2)
        nflag_bytes = s.read_hex(2) - 1 # -1 ByteNumAdjust
        flag_bytes = [s.read_hex(2) for _ in range(nflag_bytes)]
        mode = s.read_hex(2)
        if s.remaining():
            log("failed to parse signal reply completely, left: '%s'",
                s.remainder())
        return cls(
            cells=cells,
            temps=temps,
            current=current,
            total_voltage=total_voltage,
            flag_bytes=flag_bytes,
            mode=mode,
        )

    def dict_human(self, protocol: SeplosProtocol):
        """Interpret the bytes, given protocol file (mainly for flags and mode)"""

        flags: dict[str, bool] = {}
        mode: str = None

        for bit_param in protocol.signal_bit_params:
            flags[bit_param.name] = bool((self.flag_bytes[bit_param.byte_index] >> bit_param.bit_index) & 1)

        mode = protocol.signal_modes[self.mode]

        return dict(
            cells=self.cells,
            temps=self.temps,
            current=self.current,
            total_voltage=self.total_voltage,
            flags=flags,
            model=mode,
        )


class Reader:

    SEEK_START = 0
    SEEK_LEN = 1
    SEEK_END = 2

    def __init__(self, dev):
        self._fd = os.open(dev, os.O_RDONLY)
        self._abort = False

        # self._in_cmd_code = None
        self.handler = None

        self._state = self.SEEK_START
        self._len = None
        self._cmdbuf = None


    def loop(self):
        while not self._abort:
            buf = os.read(self._fd, 1024)
            if not buf:
                log("got empty read")
                break
            self._process_chunk(buf)

    def _process_chunk(self, buf):
        if isinstance(buf, bytes):
            buf = buf.decode('ascii')

        while buf:
            if self._state == self.SEEK_START:
                try:
                    idx = buf.index('~')
                    if idx != 0:
                        log("skipping %s bytes at start of buf", idx)
                    self._cmdbuf = buf[idx+1:]
                    self._state = self.SEEK_LEN
                    buf = None
                except:
                    log("skipping buf: %s", buf)
                    break
            else:
                self._cmdbuf += buf
                buf = None

            if self._state == self.SEEK_LEN:
                if len(self._cmdbuf) >= 12:
                    self._len = int(self._cmdbuf[9:12], 16)
                    self._state = self.SEEK_END

            if self._state == self.SEEK_END:
                extra = len(self._cmdbuf) - (self._len + 16)
                if extra >= 0:
                    if extra:
                        buf = self._cmdbuf[-extra:]
                        log("queueing %s bytes for next cmd", extra)
                        self._cmdbuf = self._cmdbuf[:-extra]
                    self._process_cmd(self._cmdbuf)
                    self._cmdbuf = None
                    self._len = None
                    self._state = self.SEEK_START

    def _process_cmd(self, cmdbuf):
        blen = int(cmdbuf[9:12], 16)
        blen_cs = cmdbuf[8]
        if blen_cs != len_cs(blen):
            log("len cs of incoming pkg does not match: len=%s cs=%s exp=%s",
                blen, blen_cs, fmt_cs(blen, 1))

        pkg_cs = cmdbuf[-4:]
        raw_cs = sum(ord(c) for c in cmdbuf[:-4])
        if pkg_cs != fmt_cs(raw_cs, 4):
            log("pkg cs of incoming pkg does not match: raw=%s cs=%s exp=%s",
                raw_cs, pkg_cs, fmt_cs(raw_cs, 4))

        self.handler(cmdbuf)

        sys.exit(0)

    # def read(self):
    #     ## read & discard until first char is ~

    #     # skip the ~
    #     self.skip(1)

    #     len_cs = self._buffer[8]
    #     data_len = int(self._buffer[9:12], 16)
    #     assert len_cs == fmt_cs(data_len, 1)

    #     ## accumulate data_len + 16
    #     data = self._buffer[12:-4]

    #     pkg_cs = self._buffer[-4:]
    #     raw_cs = sum(ord(c) for c in self._buffer[:-4])
    #     assert pkg_cs == fmt_cs(raw_cs, 4)


    @classmethod
    def _process_cmd_status(cls, buf):
        error_code = buf[6:8]
        if error_code != "00":
            log("got error code %s", error_code)
            return None

        data = buf[12:-4]
        s = StreamReader(data, 4)
        ncells = s.read_hex(2)
        cell_mv = [s.read_hex(4) for _ in range(ncells)]
        ntemps = s.read_hex(2)
        temp_map = lambda rt: (rt - 2731)/10
        temp_c = [temp_map(s.read_hex(4)) for _ in range(ntemps)]
        sign_map = lambda v: v if v < 2**15 else v - 2**16

        return cls(
            ncells=ncells,
            cell_mv=cell_mv,
            ntemps=ntemps,
            temp_c=temp_c,
            charge_discharge_a=sign_map(s.read_hex(4)) / 100,
            total_voltage_v=s.read_hex(4) / 100,
            residual_capacity_ah=s.read_hex(4) / 100,
            _extra=s.read_hex(2), # this is the numer of remaining fields
            batt_capacity_ah=s.read_hex(4) / 100,
            soc_percent=s.read_hex(4) / 10,
            rated_capacity_ah=s.read_hex(4) / 100,
            ncycles=s.read_hex(4),
            soh_percent=s.read_hex(4) / 10,
            port_voltage_v=s.read_hex(4) / 100,
        )

def parse_args(args):
    parser = argparse.ArgumentParser("Communicate with seplos BMS")
    parser.add_argument('--address', type=str, default='00',
                        help="BMS ID to talk to")
    parser.add_argument('--device', type=str, default='/dev/ttyUSB0',
                        help="UART device path")
    parser.add_argument('--dbg-request', action='store_true',
                        help="only print the request, don't send it")
    parser.add_argument('--protocol', type=str,
                        help="protocol XML file")
    command = parser.add_subparsers(dest='command')
    meter = command.add_parser('meter')
    signal = command.add_parser('signal')
    raw = command.add_parser('raw')
    raw.add_argument('--code', type=str, required=True,
                     help='command code (2 hex digits)')
    raw.add_argument('--data', type=str, default='',
                     help='command data (any length hex digits string)')
    download_settings = command.add_parser('download-settings')
    download_settings.add_argument('-f', '--file', required=True,
                                   help='xml file to store current BMS settings')
    upload_settings = command.add_parser('upload-settings')
    upload_settings.add_argument('-f', '--file', required=True,
                                 help='xml file with settings to upload')

    return parser.parse_args(args)


def main(args):
    opts = parse_args(args)

    code = None
    data = None
    protocol = None

    if opts.protocol:
        protocol = SeplosProtocol.from_proto_file(opts.protocol)
    if opts.command == 'meter':
        code = '42'
        data = '00'
    elif opts.command == 'signal':
        code = '44'
        data = '00'
    elif opts.command == 'raw':
        code = opts.code
        data = opts.data
    elif opts.command == 'download-settings':
        code = '47'
        data = '00'
    elif opts.command == 'upload-settings':
        code = 'A1'
        params = SeplosParams.from_save_file(protocol, opts.file)
        data = params.to_msg_body()
    else:
        raise Exception("not supported ATM")

    cmd_out = build_cmd(code=code, data=data, addr=opts.address)
    log("sending: %s", cmd_out)
    if opts.dbg_request:
        return

    def handler(cmdbuf):
        log("received: %s", cmdbuf)

        error_code = cmdbuf[6:8]
        if error_code != "00":
            log("got error code %s", error_code)
            return

        reply_data = cmdbuf[12:-4]
        if code == '42':
            parsed = MeterReply.from_reply(reply_data)
            log('%s', json.dumps(parsed.dict(), indent=2))
        elif code == '44':
            parsed = SignalReply.from_reply(reply_data)
            human = parsed.dict_human(protocol) if protocol else parsed.dict()
            log('%s', json.dumps(human, indent=2))
        elif code == '47':
            parsed = SeplosParams.from_msg_body(protocol, reply_data)
            parsed.to_save_file(opts.file)
        elif code == 'A1':
            if len(reply_data) != 0:
                log('reply len %s, expected 0', len(reply_data))
        else:
            pass

    r = Reader(opts.device)
    r.handler = handler

    t = threading.Thread(target=r.loop)
    t.start()
    time.sleep(0.2) # TODO: we can do better
    w = os.open(opts.device, os.O_WRONLY)
    os.write(w, f'~{cmd_out}\r'.encode('ascii'))
    os.close(w)
    t.join()


if __name__ == '__main__':
    main(sys.argv[1:])
