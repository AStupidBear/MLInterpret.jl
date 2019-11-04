using PyCall: python

function get_os_release(key = "ID")
    val = try
        String(read(pipeline(`cat /etc/os-release`, `grep ^$key=`, `cut -d= -f2`)))[1:end-1]
    catch
        ""
    end
    replace(val, "\"" => "")
end

function system_install(pkg)
    os = get_os_release()
    if os == "ubuntu" || os == "debian"
        pkgman = "apt"
    elseif os == "rhel" || os == "centos"
        pkgman = "yum"
    end
    if !isnothing(Sys.which("sudo"))
        run(`sudo $pkgman install -y $pkg`)
    else
        run(`$pkgman update -y`)
        run(`$pkgman install -y $pkg`)
    end
end

if isnothing(Sys.which("dot"))
    try
        system_install("graphviz")
    catch
        @warn("please install graphviz manually using system manager")
    end
end

run(`$python -m pip install pandas sklearn matplotlib lightgbm shap keras tzlocal PyPDF2 unidecode skater`)