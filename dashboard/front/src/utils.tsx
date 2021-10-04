export function zip(a:Array<any>, b:Array<any>)
{
    return a.map((k, i) => [k, b[i]])
}

export function isEmpty(obj) {
    return Object.keys(obj).length === 0;
  }